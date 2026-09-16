"""Usage accounting: what each request cost, in tokens and in attributed dollars.

Three consumers need the same numbers and must not disagree: the ``usage`` block an
OpenAI-compatible client reads off the final response, the Prometheus series a dashboard
plots, and the per-tenant running totals an operator reads out of the gateway. This module
computes them once, in :class:`UsageAccumulator`, and hands the same
:class:`UsageRecord` to all three.

Where the token counts come from, in order of preference:

1. the backend's own ``usage`` on the terminating event -- authoritative, because the
   backend tokenised the prompt and produced the completion;
2. the token ids carried by each streamed event -- exact for the in-process engine;
3. a count of non-empty text deltas -- the fallback for an HTTP backend that streams text
   and reports no usage, where one delta is one token in practice;
4. a character-based estimate of the prompt when the gateway has no tokenizer at all.

Only the last is an approximation, and a record produced that way sets
:attr:`UsageRecord.estimated`, so nothing downstream can mistake it for a measurement.

Prices are *configured list prices* from ``configs/models.yaml``, used for attributing
spend between tenants. They are an input, not a result: nothing in this module measures
what anything costs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping

    from turboserve.gateway.backends.protocol import TokenEvent
    from turboserve.gateway.metrics import GatewayMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "ModelPrice",
    "PriceTable",
    "TenantTotals",
    "UsageAccumulator",
    "UsageRecord",
    "UsageTracker",
    "estimate_text_tokens",
]

#: Average characters per token used only when no tokenizer is available. Four is the
#: common rule of thumb for English text with a byte-level BPE vocabulary; any record that
#: relies on it is flagged ``estimated``.
_CHARS_PER_TOKEN = 4

_TOKENS_PER_MILLION = 1_000_000


def estimate_text_tokens(text: str) -> int:
    """Rough token count for text the gateway cannot tokenise. At least one."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


class ModelPrice(BaseModel):
    """Configured list price for one model, per million tokens.

    ``source`` documents where the figure came from so a dashboard can say so; it defaults
    to ``"configured"`` precisely because these are not measured numbers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_1m_usd: float = Field(ge=0.0)
    output_per_1m_usd: float = Field(ge=0.0)
    currency: str = "USD"
    source: str = "configured"

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Attributed cost of one request."""
        prompt_cost = max(0, prompt_tokens) * self.input_per_1m_usd
        completion_cost = max(0, completion_tokens) * self.output_per_1m_usd
        return (prompt_cost + completion_cost) / _TOKENS_PER_MILLION


class PriceTable:
    """Model name to :class:`ModelPrice`, with a graceful miss.

    A model without a price is normal (a locally served reference engine has no list
    price), and it makes :meth:`cost_usd` return ``None`` rather than zero -- "unpriced" and
    "free" must not look the same on a bill.
    """

    __slots__ = ("_prices",)

    def __init__(self, prices: Mapping[str, ModelPrice] | None = None) -> None:
        self._prices: dict[str, ModelPrice] = dict(prices or {})

    @classmethod
    def from_mapping(cls, data: Mapping[str, Mapping[str, Any] | ModelPrice]) -> PriceTable:
        """Build from raw configuration."""
        prices = {
            name: value if isinstance(value, ModelPrice) else ModelPrice.model_validate(value)
            for name, value in data.items()
        }
        return cls(prices)

    def get(self, model: str) -> ModelPrice | None:
        """Price for ``model``, or ``None`` when it is not priced."""
        return self._prices.get(model)

    def cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """Attributed cost, or ``None`` when the model has no configured price."""
        price = self._prices.get(model)
        if price is None:
            return None
        return price.cost_usd(prompt_tokens, completion_tokens)

    def models(self) -> list[str]:
        """Priced model names, sorted."""
        return sorted(self._prices)

    def __contains__(self, model: object) -> bool:
        return model in self._prices

    def __len__(self) -> int:
        return len(self._prices)

    def __repr__(self) -> str:
        return f"PriceTable({', '.join(sorted(self._prices))})"


@dataclass(slots=True)
class UsageRecord:
    """The finished accounting of one request.

    ``ttft_s``/``tpot_s``/``e2e_s`` follow the definitions in ``docs/benchmarking.md`` and
    :class:`turboserve.engine.core.types.RequestTiming`: TTFT is arrival to first output
    token, TPOT is the mean gap between output tokens after the first, E2E is arrival to
    finish. All three are ``None`` when the request did not get far enough to have them.
    """

    request_id: str
    tenant_id: str
    model: str
    backend: str = ""
    lane: str = "stable"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    ttft_s: float | None = None
    tpot_s: float | None = None
    e2e_s: float | None = None
    status: str = "ok"
    finish_reason: str | None = None
    cost_usd: float | None = None
    estimated: bool = False
    """True when the prompt token count is a character-based estimate, not a real count."""

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def ok(self) -> bool:
        """Whether the request completed normally."""
        return self.status == "ok"

    def to_openai_usage(self) -> dict[str, Any]:
        """The ``usage`` object of an OpenAI response body.

        ``prompt_tokens_details.cached_tokens`` is the field OpenAI uses for prompt-cache
        hits, and it is exactly what the engine's prefix cache reports, so a client that
        already understands prompt caching needs no turboserve-specific code.
        """
        usage: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_prompt_tokens:
            usage["prompt_tokens_details"] = {"cached_tokens": self.cached_prompt_tokens}
        return usage

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for logs and for the chaos/bench result files."""
        return {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "model": self.model,
            "backend": self.backend,
            "lane": self.lane,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "e2e_s": self.e2e_s,
            "status": self.status,
            "finish_reason": self.finish_reason,
            "cost_usd": self.cost_usd,
            "estimated": self.estimated,
        }


class UsageAccumulator:
    """Accumulates one in-flight request's timings and token counts.

    Fed every event the router yields. The clock is ``time.perf_counter`` by default and
    injectable, and ``arrival_ts`` comes from the request rather than from construction so
    that TTFT includes the time the request spent in auth, quota checks and routing -- the
    number a client would measure, not the number that flatters the engine.
    """

    __slots__ = (
        "_clock",
        "_completion_tokens",
        "_counted_deltas",
        "_estimated",
        "_finish_reason",
        "arrival_ts",
        "backend",
        "cached_prompt_tokens",
        "lane",
        "model",
        "prompt_tokens",
        "request_id",
        "t_finish",
        "t_first_token",
        "tenant_id",
    )

    def __init__(
        self,
        request_id: str,
        tenant_id: str,
        model: str,
        *,
        backend: str = "",
        lane: str = "stable",
        prompt_tokens: int = 0,
        estimated: bool = False,
        arrival_ts: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.request_id = request_id
        self.tenant_id = tenant_id
        self.model = model
        self.backend = backend
        self.lane = lane
        self.prompt_tokens = prompt_tokens
        self.cached_prompt_tokens = 0
        self._clock = clock
        self.arrival_ts = clock() if arrival_ts is None else arrival_ts
        self.t_first_token: float | None = None
        self.t_finish: float | None = None
        self._completion_tokens = 0
        self._counted_deltas = 0
        self._finish_reason: str | None = None
        self._estimated = estimated

    # -- feeding ----------------------------------------------------------------------

    def on_event(self, event: TokenEvent, *, now: float | None = None) -> None:
        """Fold one streamed event into the accounting.

        Token ids are counted when present; a text-only delta counts as one token, which is
        what an HTTP backend that reports no usage gives us. A terminating event carrying a
        ``usage`` block overrides both, because the producer knows better than we do.
        """
        stamp = self._clock() if now is None else now
        produced = event.num_tokens or (1 if event.text else 0)
        if produced and self.t_first_token is None:
            self.t_first_token = stamp
        self._completion_tokens += event.num_tokens
        self._counted_deltas += produced
        if event.finished:
            self.t_finish = stamp
            if event.finish_reason is not None:
                self._finish_reason = str(event.finish_reason)
            if event.usage:
                self._absorb_usage(event.usage)

    def _absorb_usage(self, usage: Mapping[str, Any]) -> None:
        """Take the backend's authoritative counts over our own tally."""
        prompt = usage.get("prompt_tokens")
        if isinstance(prompt, int) and prompt >= 0:
            self.prompt_tokens = prompt
            self._estimated = False
        completion = usage.get("completion_tokens")
        if isinstance(completion, int) and completion >= 0:
            self._completion_tokens = completion
            self._counted_deltas = completion
        cached = usage.get("cached_prompt_tokens")
        if isinstance(cached, int) and cached >= 0:
            self.cached_prompt_tokens = cached

    # -- derived values ---------------------------------------------------------------

    @property
    def completion_tokens(self) -> int:
        """Output tokens so far, falling back to the delta count for text-only streams."""
        return self._completion_tokens or self._counted_deltas

    @property
    def ttft_s(self) -> float | None:
        """Arrival to first output token."""
        if self.t_first_token is None:
            return None
        return self.t_first_token - self.arrival_ts

    @property
    def e2e_s(self) -> float | None:
        """Arrival to the final event."""
        if self.t_finish is None:
            return None
        return self.t_finish - self.arrival_ts

    @property
    def tpot_s(self) -> float | None:
        """Mean gap between output tokens after the first.

        ``None`` below two output tokens: a single-token completion has no inter-token gap,
        and dividing by zero output intervals would report a made-up number.
        """
        if self.t_first_token is None or self.t_finish is None:
            return None
        n_out = self.completion_tokens
        if n_out < 2:
            return None
        return (self.t_finish - self.t_first_token) / (n_out - 1)

    def finish(
        self,
        *,
        status: str = "ok",
        finish_reason: str | None = None,
        now: float | None = None,
    ) -> UsageRecord:
        """Close the accounting and return the immutable record."""
        if self.t_finish is None:
            self.t_finish = self._clock() if now is None else now
        return UsageRecord(
            request_id=self.request_id,
            tenant_id=self.tenant_id,
            model=self.model,
            backend=self.backend,
            lane=self.lane,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
            ttft_s=self.ttft_s,
            tpot_s=self.tpot_s,
            e2e_s=self.e2e_s,
            status=status,
            finish_reason=finish_reason or self._finish_reason,
            estimated=self._estimated,
        )


@dataclass(slots=True)
class TenantTotals:
    """Running totals for one tenant since the process started."""

    tenant_id: str
    requests: int = 0
    failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens across every request."""
        return self.prompt_tokens + self.completion_tokens

    def add(self, record: UsageRecord) -> None:
        """Fold a finished record in."""
        self.requests += 1
        if not record.ok:
            self.failed += 1
        self.prompt_tokens += record.prompt_tokens
        self.completion_tokens += record.completion_tokens
        self.cached_prompt_tokens += record.cached_prompt_tokens
        if record.cost_usd:
            self.cost_usd += record.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping for JSON output."""
        return {
            "tenant_id": self.tenant_id,
            "requests": self.requests,
            "failed": self.failed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


class UsageTracker:
    """Turns finished accumulators into records, metrics and per-tenant totals.

    One instance per gateway app. It is the only place that knows about all three
    destinations, which is what keeps the ``usage`` a client sees, the Prometheus series and
    the operator's totals from drifting apart.
    """

    __slots__ = ("_metrics", "_prices", "_totals")

    def __init__(
        self,
        *,
        prices: PriceTable | None = None,
        metrics: GatewayMetrics | None = None,
    ) -> None:
        self._prices = prices if prices is not None else PriceTable()
        self._metrics = metrics
        self._totals: dict[str, TenantTotals] = {}

    @property
    def prices(self) -> PriceTable:
        """The configured price table used for attribution."""
        return self._prices

    def start(
        self,
        *,
        request_id: str,
        tenant_id: str,
        model: str,
        prompt_tokens: int = 0,
        estimated: bool = False,
        arrival_ts: float | None = None,
        backend: str = "",
        lane: str = "stable",
    ) -> UsageAccumulator:
        """Open the accounting for a request."""
        return UsageAccumulator(
            request_id,
            tenant_id,
            model,
            backend=backend,
            lane=lane,
            prompt_tokens=prompt_tokens,
            estimated=estimated,
            arrival_ts=arrival_ts,
        )

    def complete(
        self,
        accumulator: UsageAccumulator,
        *,
        status: str = "ok",
        finish_reason: str | None = None,
    ) -> UsageRecord:
        """Close an accumulator, price it, publish it and return the record."""
        record = accumulator.finish(status=status, finish_reason=finish_reason)
        return self.record(record)

    def record(self, record: UsageRecord) -> UsageRecord:
        """Price, publish and total an already-built record."""
        if record.cost_usd is None:
            record.cost_usd = self._prices.cost_usd(
                record.model, record.prompt_tokens, record.completion_tokens
            )
        totals = self._totals.get(record.tenant_id)
        if totals is None:
            totals = TenantTotals(tenant_id=record.tenant_id)
            self._totals[record.tenant_id] = totals
        totals.add(record)
        if self._metrics is not None:
            self._metrics.record_request(
                tenant=record.tenant_id,
                model=record.model,
                backend=record.backend,
                lane=record.lane,
                status=record.status,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                ttft_s=record.ttft_s,
                tpot_s=record.tpot_s,
                e2e_s=record.e2e_s,
                cost_usd=record.cost_usd,
            )
        return record

    def totals(self) -> dict[str, TenantTotals]:
        """Per-tenant running totals, keyed by tenant id."""
        return dict(self._totals)

    def totals_for(self, tenant_id: str) -> TenantTotals:
        """Running totals for one tenant; zeros when it has sent nothing."""
        return self._totals.get(tenant_id) or TenantTotals(tenant_id=tenant_id)

    def reset(self) -> None:
        """Drop all totals. Used by tests."""
        self._totals.clear()
