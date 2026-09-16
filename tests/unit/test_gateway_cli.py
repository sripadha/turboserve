"""The ``turboserve gateway`` sub-app, and the standalone mock server it can front.

``serve`` itself is not invoked (it blocks in uvicorn); its router-construction step is
tested directly, and the mock server is exercised in-process through a test client rather
than as a subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from turboserve.config import Settings
from turboserve.gateway.app import gateway_app, router_for_engine
from turboserve.gateway.auth import hash_api_key
from turboserve.gateway.backends.mock import MockBackend, build_mock_app
from turboserve.gateway.backends.openai_compat import OpenAICompatBackend

runner = CliRunner()

REPO = Path(__file__).resolve().parents[2]


def test_hash_key_prints_the_digest_tenants_yaml_wants() -> None:
    result = runner.invoke(gateway_app, ["hash-key", "sk-example"])
    assert result.exit_code == 0
    assert result.stdout.strip() == hash_api_key("sk-example")


def test_config_check_accepts_the_shipped_configuration() -> None:
    result = runner.invoke(
        gateway_app,
        [
            "config-check",
            "--tenants",
            str(REPO / "configs" / "tenants.yaml"),
            "--models",
            str(REPO / "configs" / "models.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "acme" in result.stdout
    assert "mock-model" in result.stdout


def test_config_check_reports_a_broken_tenants_file(tmp_path: Path) -> None:
    bad = tmp_path / "tenants.yaml"
    bad.write_text("tenants: []\n", encoding="utf-8")
    result = runner.invoke(
        gateway_app,
        ["config-check", "--tenants", str(bad), "--models", str(REPO / "configs" / "models.yaml")],
    )
    assert result.exit_code != 0


def test_config_check_reports_a_broken_models_file(tmp_path: Path) -> None:
    bad = tmp_path / "models.yaml"
    bad.write_text("models:\n  - name: m\n", encoding="utf-8")
    result = runner.invoke(
        gateway_app,
        ["config-check", "--tenants", str(REPO / "configs" / "tenants.yaml"), "--models", str(bad)],
    )
    assert result.exit_code != 0


def test_engine_mock_builds_an_in_process_pool(tmp_path: Path) -> None:
    settings = Settings(model="tiny", models_file=tmp_path / "absent.yaml")
    router = router_for_engine("mock", settings, tmp_path / "absent.yaml")
    assert router.models() == ["tiny"]
    assert isinstance(router.pool("tiny").entries[0].backend, MockBackend)


def test_engine_url_builds_an_openai_compatible_pool(tmp_path: Path) -> None:
    settings = Settings(model="tiny", models_file=tmp_path / "absent.yaml")
    router = router_for_engine("http://vllm:8000/v1", settings, tmp_path / "absent.yaml")
    backend = router.pool("tiny").entries[0].backend
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.base_url == "http://vllm:8000/v1"


def test_engine_config_reads_the_models_file(tmp_path: Path) -> None:
    models = tmp_path / "models.yaml"
    models.write_text(
        "models:\n"
        "  - name: from-file\n"
        "    backends:\n"
        "      - name: a\n"
        "        backend: mock\n"
        "        options: {models: [from-file]}\n",
        encoding="utf-8",
    )
    router = router_for_engine("config", Settings(models_file=models), models)
    assert router.models() == ["from-file"]


def test_an_unknown_engine_is_rejected(tmp_path: Path) -> None:
    from typer import BadParameter

    settings = Settings(models_file=tmp_path / "absent.yaml")
    with pytest.raises(BadParameter, match="must be 'config', 'mock'"):
        router_for_engine("sqlite://whatever", settings, tmp_path / "absent.yaml")


def test_mock_server_is_the_real_gateway_with_auth_off() -> None:
    # The chaos harness launches this as a subprocess; running it in-process here proves the
    # faults it injects travel the same auth, routing and SSE code a real request does.
    app = build_mock_app(models=["mock-model"], max_tokens=3, name="replica-1")
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/v1/models").json()["data"][0]["id"] == "mock-model"
        response = client.post(
            "/v1/completions", json={"model": "mock-model", "prompt": "hi", "stream": True}
        )
        assert response.status_code == 200
        assert response.text.endswith("data: [DONE]\n\n")
        assert app.state.mock_backend.stats.completed == 1


def test_mock_server_faults_are_configurable() -> None:
    app = build_mock_app(models=["mock-model"], error_probability=1.0)
    with TestClient(app) as client:
        response = client.post("/v1/completions", json={"model": "mock-model", "prompt": "hi"})
    assert response.status_code == 503
