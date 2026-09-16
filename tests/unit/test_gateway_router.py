"""Routing: lane weighting, health, retry-before-first-byte and the concurrency gate."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from turboserve.engine.core.types import FinishReason
from turboserve.gateway.backends.mock import MockBackend
from turboserve.gateway.backends.protocol import (
    BackendOverloadedError,
    BackendRequestError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
    NonRetryableBackendError,
    TokenEvent,
)
from turboserve.gateway.limits import LimiterRegistry, RateLimitExceeded
from turboserve.gateway.router import (
    CanaryWeightSource,
    ModelsFile,
    RoutedEvent,
    Router,
    RouterConfigError,
    build_router,
)
from turboserve.gateway.tenants import Tenant, TenantRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def make_request(**overrides: object) -> GenerateRequest:
    """A minimal valid generate request."""
    data: dict[str, object] = {
        "request_id": "req-1",
        "tenant_id": "acme",
        "model": "m",
        "prompt": "hello",
    }
    data.update(overrides)
    return GenerateRequest.model_validate(data)


async def route(router: Router, req: GenerateRequest | None = None) -> list[RoutedEvent]:
    """Collect everything the router yields for one request."""
    return [event async for event in router.generate(req or make_request())]


class ScriptedBackend:
    """A backend that fails in a specified way at a specified point in the stream."""

    supports_lora = False

    def __init__(
        self,
        name: str,
        *,
        fail_with: Exception | None = None,
        after_tokens: int = 0,
        tokens: int = 2,
        healthy: bool = True,
    ) -> None:
        self.name = name
        self.fail_with = fail_with
        self.after_tokens = after_tokens
        self.tokens = tokens
        self._healthy = healthy
        self.calls = 0
        self.closed = False

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        self.calls += 1
        for index in range(self.tokens):
            if self.fail_with is not None and index == self.after_tokens:
                raise self.fail_with
            yield TokenEvent.delta(req.request_id, [index], f"t{index}")
        if self.fail_with is not None and self.tokens == self.after_tokens:
            raise self.fail_with
        yield TokenEvent.final(req.request_id, FinishReason.STOP)

    async def health(self) -> bool:
        return self._healthy

    async def models(self) -> list[str]:
        return ["m"]

    async def close(self) -> None:
        self.closed = True


class FirstReplicaRng(random.Random):
    """A ``Random`` that always draws 0.0, so selection is by pool order.

    Retry semantics must be asserted exactly, not statistically: with this RNG the first
    healthy candidate always wins and "which replica ran" becomes deterministic.
    """

    def random(self) -> float:
        return 0.0


class FakeCanary:
    """The slice of the canary controller the router depends on."""

    def __init__(self, weight: float = 0.0) -> None:
        self.canary_weight = weight
        self.observations: list[tuple[str, bool, float | None, float | None]] = []

    def observe(
        self, lane: str, ok: bool, ttft_ms: float | None, e2e_ms: float | None, now: float
    ) -> None:
        self.observations.append((lane, ok, ttft_ms, e2e_ms))


# -- pools and selection ----------------------------------------------------------------


def test_unknown_model_is_a_model_not_found_error() -> None:
    with pytest.raises(ModelNotFoundError):
        Router().pool("nope")


def test_duplicate_backend_name_in_one_pool_is_refused() -> None:
    router = Router()
    router.add_backend("m", MockBackend(name="a", models=["m"]))
    with pytest.raises(RouterConfigError, match="already in the pool"):
        router.add_backend("m", MockBackend(name="a", models=["m"]))


def test_models_are_listed_sorted() -> None:
    router = Router()
    router.add_backend("z", MockBackend(name="a", models=["z"]))
    router.add_backend("a", MockBackend(name="b", models=["a"]))
    assert router.models() == ["a", "z"]
    assert router.has_model("z") and not router.has_model("q")


def test_canary_fraction_is_a_clamped_percentage() -> None:
    router = Router()
    assert router.canary_fraction() == 0.0  # no controller means no canary traffic
    canary = FakeCanary(25.0)
    router.set_canary(canary)
    assert router.canary_fraction() == pytest.approx(0.25)
    canary.canary_weight = 400.0
    assert router.canary_fraction() == 1.0
    canary.canary_weight = -5.0
    assert router.canary_fraction() == 0.0


def test_lane_draw_follows_the_canary_weight_statistically() -> None:
    router = Router(rng=random.Random(20260916), canary=FakeCanary(25.0))
    router.add_backend("m", MockBackend(name="stable", models=["m"]))
    router.add_backend("m", MockBackend(name="canary", models=["m"]), lane="canary")
    pool = router.pool("m")
    draws = [router.choose_lane(pool) for _ in range(4000)]
    share = draws.count("canary") / len(draws)
    assert share == pytest.approx(0.25, abs=0.03)


def test_lane_draw_skips_a_lane_with_no_healthy_replica() -> None:
    router = Router(rng=random.Random(1), canary=FakeCanary(100.0))
    router.add_backend("m", MockBackend(name="stable", models=["m"]))
    canary_entry = router.add_backend("m", MockBackend(name="canary", models=["m"]), lane="canary")
    canary_entry.healthy = False
    pool = router.pool("m")
    assert {router.choose_lane(pool) for _ in range(50)} == {"stable"}


async def test_weight_splits_traffic_within_a_lane() -> None:
    router = Router(rng=random.Random(7))
    router.add_backend("m", MockBackend(name="big", models=["m"]), weight=3.0)
    router.add_backend("m", MockBackend(name="small", models=["m"]), weight=1.0)
    picks = [(await router.select("m")).name for _ in range(2000)]
    assert picks.count("big") / len(picks) == pytest.approx(0.75, abs=0.03)


def test_zero_or_negative_weight_is_refused() -> None:
    router = Router()
    with pytest.raises(ValueError, match="weight must be positive"):
        router.add_backend("m", MockBackend(name="a", models=["m"]), weight=0.0)


# -- health -----------------------------------------------------------------------------


async def test_health_verdict_is_cached_for_its_ttl() -> None:
    clock = [0.0]
    backend = MockBackend(name="a", models=["m"])
    router = Router(health_ttl_s=5.0, clock=lambda: clock[0])
    entry = router.add_backend("m", backend)
    assert await router.refresh_health(entry) is True

    backend.set_healthy(False)
    assert await router.refresh_health(entry) is True  # still inside the ttl
    clock[0] = 6.0
    assert await router.refresh_health(entry) is False


async def test_a_health_probe_that_raises_counts_as_unhealthy() -> None:
    class Raising(ScriptedBackend):
        async def health(self) -> bool:
            raise RuntimeError("probe exploded")

    router = Router()
    entry = router.add_backend("m", Raising("boom"))
    assert await router.refresh_health(entry) is False


async def test_ready_requires_one_healthy_replica_per_model() -> None:
    router = Router()
    assert await router.ready() is False  # a router with no pools serves nothing
    good = MockBackend(name="a", models=["m"])
    router.add_backend("m", good)
    assert await router.ready() is True
    good.set_healthy(False)
    router._pools["m"].entries[0].checked_at = float("-inf")  # noqa: SLF001 - force a probe
    assert await router.ready() is False


async def test_health_report_names_every_replica() -> None:
    router = Router()
    router.add_backend("m", MockBackend(name="a", models=["m"]))
    router.add_backend("m", MockBackend(name="b", models=["m"]), lane="canary")
    assert await router.health_report() == {"m": {"a": True, "b": True}}


# -- retry semantics --------------------------------------------------------------------


async def test_a_retryable_failure_before_the_first_event_moves_to_another_replica() -> None:
    bad = ScriptedBackend("bad", fail_with=BackendUnavailableError("refused"), after_tokens=0)
    good = ScriptedBackend("good", tokens=2)
    router = Router(rng=FirstReplicaRng(), max_attempts=3)
    router.add_backend("m", bad)
    router.add_backend("m", good)

    events = await route(router)
    assert [event.backend for event in events] == ["good", "good", "good"]
    assert bad.calls == 1 and good.calls == 1
    # The replica that failed is taken out of rotation until its next probe.
    assert router.pool("m").entries[0].healthy is False


async def test_a_failure_after_the_first_event_is_never_retried() -> None:
    # The client already holds part of a completion: re-running it elsewhere would either
    # duplicate or contradict what they have.
    flaky = ScriptedBackend(
        "flaky", fail_with=BackendOverloadedError("died"), after_tokens=1, tokens=4
    )
    spare = ScriptedBackend("spare", tokens=4)
    router = Router(rng=FirstReplicaRng(), max_attempts=3)
    router.add_backend("m", flaky)
    router.add_backend("m", spare)

    events = await route(router)
    assert {event.backend for event in events} == {"flaky"}
    assert spare.calls == 0
    assert events[-1].is_error
    assert events[-1].event.finish_reason is FinishReason.ABORT
    assert len(events) == 2  # one real token, then the terminating failure


async def test_a_non_retryable_failure_is_raised_without_a_second_attempt() -> None:
    bad = ScriptedBackend("bad", fail_with=BackendRequestError("bad sampling params"))
    spare = ScriptedBackend("spare")
    router = Router(rng=FirstReplicaRng(), max_attempts=3)
    router.add_backend("m", bad)
    router.add_backend("m", spare)

    with pytest.raises(BackendRequestError):
        await route(router)
    assert spare.calls == 0
    # A rejected request says nothing about the replica's health.
    assert router.pool("m").entries[0].healthy is True


async def test_attempts_are_bounded_and_the_last_error_surfaces() -> None:
    first = ScriptedBackend("one", fail_with=BackendUnavailableError("down"))
    second = ScriptedBackend("two", fail_with=BackendUnavailableError("down"))
    router = Router(rng=FirstReplicaRng(), max_attempts=2)
    router.add_backend("m", first)
    router.add_backend("m", second)

    with pytest.raises(BackendUnavailableError):
        await route(router)
    assert first.calls == 1 and second.calls == 1


async def test_an_unexpected_backend_exception_becomes_a_backend_error() -> None:
    # A bug in a backend must produce a 502, not a 500 with a stack trace in the response.
    broken = ScriptedBackend("broken", fail_with=ZeroDivisionError("bug"))
    router = Router(rng=FirstReplicaRng())
    router.add_backend("m", broken)
    with pytest.raises(NonRetryableBackendError) as exc:
        await route(router)
    assert "broken failed" in str(exc.value)
    assert exc.value.__cause__.__class__ is ZeroDivisionError


async def test_no_healthy_replica_raises_before_anything_is_yielded() -> None:
    backend = MockBackend(name="a", models=["m"])
    backend.set_healthy(False)
    router = Router()
    router.add_backend("m", backend)
    with pytest.raises(BackendUnavailableError, match="no healthy backend"):
        await route(router)


def test_max_attempts_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        Router(max_attempts=0)


# -- concurrency gate and canary feedback -----------------------------------------------


async def test_the_tenant_concurrency_gate_refuses_before_the_first_event() -> None:
    tenants = TenantRegistry([Tenant(tenant_id="acme", max_concurrency=1)])
    limiters = LimiterRegistry(tenants)
    router = Router(limiters=limiters)
    router.add_backend("m", MockBackend(name="a", models=["m"], max_tokens=4))

    held = router.generate(make_request(request_id="first"))
    await anext(held)  # holds the only slot
    with pytest.raises(RateLimitExceeded) as exc:
        await route(router, make_request(request_id="second"))
    assert exc.value.limit == "concurrency"
    await held.aclose()
    # Closing the stream returns the slot, so the next request is admitted.
    assert await route(router, make_request(request_id="third"))


async def test_queue_depth_reflects_held_slots() -> None:
    tenants = TenantRegistry([Tenant(tenant_id="acme", max_concurrency=4)])
    router = Router(limiters=LimiterRegistry(tenants))
    router.add_backend("m", MockBackend(name="a", models=["m"], max_tokens=4))
    assert router.queue_depth() == 0
    stream = router.generate(make_request())
    await anext(stream)
    assert router.queue_depth() == 1
    await stream.aclose()
    assert router.queue_depth() == 0


async def test_finished_requests_are_reported_to_the_canary_controller() -> None:
    canary = FakeCanary(100.0)
    router = Router(canary=canary, rng=FirstReplicaRng())
    router.add_backend("m", MockBackend(name="c", models=["m"], max_tokens=2), lane="canary")
    await route(router)
    assert len(canary.observations) == 1
    lane, ok, ttft_ms, e2e_ms = canary.observations[0]
    assert lane == "canary" and ok is True
    assert ttft_ms is not None and e2e_ms is not None and e2e_ms >= ttft_ms


async def test_a_mid_stream_failure_is_reported_to_the_canary_as_a_failure() -> None:
    """A backend that *yields* a terminating error is not a success.

    After the first byte a backend reports failure by yielding
    ``TokenEvent.failure(...)`` rather than raising -- the stream then ends normally, and the
    obvious ``else:`` branch would feed it to the gate as a completed request. A canary
    watching in-process observations would then be blind to exactly the failure mode that
    matters most: the one the client already started reading.
    """
    canary = FakeCanary(100.0)
    router = Router(canary=canary, rng=FirstReplicaRng())
    # drop_probability=1 cuts the stream after at least one token and before the last.
    router.add_backend(
        "m",
        MockBackend(name="c", models=["m"], max_tokens=6, drop_probability=1.0),
        lane="canary",
    )
    events = [routed async for routed in router.generate(make_request())]
    assert events[-1].is_error
    assert [obs[1] for obs in canary.observations] == [False]


async def test_an_unexpected_backend_exception_after_the_first_byte_is_observed() -> None:
    """The same rule for a backend that crashes rather than failing politely."""
    canary = FakeCanary(0.0)
    router = Router(canary=canary, rng=FirstReplicaRng())
    router.add_backend(
        "m",
        ScriptedBackend("a", fail_with=ZeroDivisionError("backend bug"), after_tokens=1),
    )
    events = [routed async for routed in router.generate(make_request())]
    assert events[-1].is_error
    assert [obs[1] for obs in canary.observations] == [False]


async def test_a_controller_that_raises_does_not_break_serving() -> None:
    class Angry(FakeCanary):
        def observe(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("controller bug")

    router = Router(canary=Angry(0.0), rng=FirstReplicaRng())
    router.add_backend("m", MockBackend(name="a", models=["m"], max_tokens=2))
    assert await route(router)


# -- configuration ----------------------------------------------------------------------


def test_build_router_from_the_shipped_example_models_file() -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
    config = ModelsFile.from_yaml(path)
    assert "mock-model" in config.model_names()
    assert config.price_table().cost_usd("Qwen/Qwen2.5-7B-Instruct", 1_000_000, 0) is not None


def test_build_router_instantiates_backends_by_name(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n"
        "  - name: m\n"
        "    backends:\n"
        "      - name: a\n"
        "        backend: mock\n"
        "        options: {models: [m]}\n"
        "      - name: b\n"
        "        backend: mock\n"
        "        lane: canary\n"
        "        weight: 2.0\n"
        "        options: {models: [m]}\n",
        encoding="utf-8",
    )
    router = build_router(ModelsFile.from_yaml(path))
    pool = router.pool("m")
    assert [entry.name for entry in pool.entries] == ["a", "b"]
    assert pool.lane("canary")[0].weight == 2.0


def test_build_router_reports_an_unknown_backend_type(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  - name: m\n    backends:\n      - name: a\n        backend: nope\n",
        encoding="utf-8",
    )
    with pytest.raises(RouterConfigError, match="nope"):
        build_router(ModelsFile.from_yaml(path))


def test_build_router_reports_bad_backend_options(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n"
        "  - name: m\n"
        "    backends:\n"
        "      - name: a\n"
        "        backend: mock\n"
        "        options: {no_such_option: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(RouterConfigError, match="could not be constructed"):
        build_router(ModelsFile.from_yaml(path))


def test_models_file_errors_are_reported_with_the_path(tmp_path: Path) -> None:
    with pytest.raises(RouterConfigError, match="cannot read"):
        ModelsFile.from_yaml(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("models:\n  - name: m\n", encoding="utf-8")  # no backends
    with pytest.raises(RouterConfigError, match="invalid models file"):
        ModelsFile.from_yaml(bad)


async def test_close_closes_every_backend_once() -> None:
    shared = ScriptedBackend("shared")
    router = Router()
    router.add_backend("m1", shared)
    router.add_backend("m2", shared)
    assert router.backends() == [shared]
    await router.close()
    assert shared.closed is True


def test_attach_does_not_overwrite_existing_collaborators() -> None:
    tenants = TenantRegistry([Tenant(tenant_id="acme")])
    first = LimiterRegistry(tenants)
    router = Router(limiters=first)
    router.attach(limiters=LimiterRegistry(tenants))
    router.add_backend("m", MockBackend(name="a", models=["m"]))
    assert router._limiters is first  # noqa: SLF001 - asserting the wiring rule


def test_the_real_canary_controller_satisfies_the_routers_protocol() -> None:
    """The one place the two modules meet, pinned in both directions.

    The gateway deliberately never imports :mod:`turboserve.canary`; it declares
    ``CanaryWeightSource`` and trusts the controller to match. Nothing else checks that,
    so this test does: it imports the real controller, asserts the structural check passes,
    and drives the router through it. A change to either side that breaks the pairing fails
    here rather than in a deployment with a rollout in progress.
    """
    from turboserve.canary.controller import CanaryController

    controller = CanaryController()
    assert isinstance(controller, CanaryWeightSource)

    router = Router(canary=controller)
    assert router.canary_fraction() == 0.0  # IDLE: everything goes to stable

    controller.start("v2", now=0.0)
    assert controller.canary_weight > 0
    assert router.canary_fraction() == controller.canary_weight / 100.0

    # The router calls observe() with the lane that actually served the request; the
    # controller must accept that call shape with positional arguments.
    controller.observe("canary", True, 12.0, 100.0, 1.0)
    assert controller.summary("canary", now=1.0).requests == 1
