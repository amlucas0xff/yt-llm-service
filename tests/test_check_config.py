"""Resolved Compose GPU policy, including safe diagnostics."""

import copy
import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "check_config", Path(__file__).resolve().parents[1] / "scripts" / "check_config.py"
)
check_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_config)


@pytest.fixture
def resolved():
    return {
        "services": {
            "yt-llm-service": {
                "environment": {"SIDECAR_GPU_SEPARATE": "false", "LLAMA_CPP_IDLE_SECONDS": "5", "HF_TOKEN": "secret-do-not-print"},
                "deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "capabilities": ["gpu"], "device_ids": ["0"]}]}}},
            },
            "llama-cpp": {
                "environment": {"LLAMA_CPP_IDLE_SECONDS": "5"},
                "deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "capabilities": ["gpu"], "device_ids": ["0"]}]}}},
            },
        }
    }


@pytest.mark.parametrize("idle", ["5", "0"])
def test_shared_gpu_accepts_idle_before_wait(resolved, idle):
    resolved["services"]["llama-cpp"]["environment"]["LLAMA_CPP_IDLE_SECONDS"] = idle
    resolved["services"]["yt-llm-service"]["environment"]["LLAMA_CPP_IDLE_SECONDS"] = idle
    assert "shared GPU" in check_config.check(resolved)


@pytest.mark.parametrize("idle", ["300", "-1", "30"])
def test_shared_gpu_rejects_idle_outside_wait(resolved, idle):
    resolved["services"]["llama-cpp"]["environment"]["LLAMA_CPP_IDLE_SECONDS"] = idle
    resolved["services"]["yt-llm-service"]["environment"]["LLAMA_CPP_IDLE_SECONDS"] = idle
    with pytest.raises(check_config.ConfigError, match="Lower the idle interval"):
        check_config.check(resolved)


def test_split_gpu_accepts_disabled_idle_and_never_waits(resolved):
    resolved["services"]["yt-llm-service"]["environment"]["SIDECAR_GPU_SEPARATE"] = "true"
    resolved["services"]["llama-cpp"]["environment"]["LLAMA_CPP_IDLE_SECONDS"] = "-1"
    resolved["services"]["yt-llm-service"]["environment"]["LLAMA_CPP_IDLE_SECONDS"] = "-1"
    resolved["services"]["llama-cpp"]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] = ["1"]
    assert "separate GPUs" in check_config.check(resolved)


@pytest.mark.parametrize("api_reservation,sidecar_reservation", [
    ({"count": 1}, {"device_ids": ["1"]}),
    ({"driver": "nvidia", "capabilities": ["gpu"], "device_ids": ["0"]},
     {"driver": "nvidia", "capabilities": ["gpu"], "device_ids": ["0"]}),
])
def test_split_marker_requires_proven_distinct_reservations(resolved, api_reservation, sidecar_reservation):
    resolved["services"]["yt-llm-service"]["environment"]["SIDECAR_GPU_SEPARATE"] = "true"
    resolved["services"]["yt-llm-service"]["deploy"]["resources"]["reservations"]["devices"] = [api_reservation]
    resolved["services"]["llama-cpp"]["deploy"]["resources"]["reservations"]["devices"] = [sidecar_reservation]
    with pytest.raises(check_config.ConfigError):
        check_config.check(resolved)


def test_smoke_uses_only_explicit_api_gpu_ids(resolved):
    assert check_config.smoke_gpu_device(resolved) == "device=0"
    api = resolved["services"]["yt-llm-service"]
    sidecar = resolved["services"]["llama-cpp"]
    api["environment"]["SIDECAR_GPU_SEPARATE"] = "true"
    sidecar["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] = ["1"]
    sidecar["environment"]["LLAMA_CPP_IDLE_SECONDS"] = "-1"
    api["environment"]["LLAMA_CPP_IDLE_SECONDS"] = "-1"
    assert check_config.smoke_gpu_device(resolved) == "device=0"
    api["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] = ["1"]
    sidecar["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] = ["0"]
    assert check_config.smoke_gpu_device(resolved) == "device=1"


def test_smoke_quotes_multiple_gpu_ids_for_docker_csv(resolved):
    for service in resolved["services"].values():
        service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] = ["0", "1"]
    assert check_config.smoke_gpu_device(resolved) == '"device=0,1"'


@pytest.mark.parametrize("reservation", [{"count": 1}, {"device_ids": []}, {"device_ids": ["0,1"]}])
def test_smoke_rejects_ambiguous_or_unsafe_api_gpu_ids(resolved, reservation):
    resolved["services"]["yt-llm-service"]["deploy"]["resources"]["reservations"]["devices"] = [reservation]
    with pytest.raises(check_config.ConfigError, match="GPU reservation"):
        check_config.smoke_gpu_device(resolved)


def test_smoke_rejects_mismatched_shared_gpu(resolved):
    resolved["services"]["llama-cpp"]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] = ["1"]
    with pytest.raises(check_config.ConfigError, match="shared GPU"):
        check_config.smoke_gpu_device(resolved)


def test_mismatched_idle_fails_closed_without_printing_secrets(resolved):
    resolved["services"]["llama-cpp"]["environment"]["LLAMA_CPP_IDLE_SECONDS"] = "300"
    with pytest.raises(check_config.ConfigError, match="same LLAMA_CPP_IDLE_SECONDS") as error:
        check_config.check(resolved)
    assert "secret-do-not-print" not in str(error.value)


def test_invalid_marker_fails_closed(resolved):
    resolved["services"]["yt-llm-service"]["environment"]["SIDECAR_GPU_SEPARATE"] = "yes"
    with pytest.raises(check_config.ConfigError):
        check_config.check(resolved)


def test_compose_errors_and_secrets_never_reach_output(resolved, capsys):
    import json

    payload = copy.deepcopy(resolved)
    payload["services"]["llama-cpp"]["environment"]["LLAMA_CPP_IDLE_SECONDS"] = "300"
    payload["services"]["yt-llm-service"]["environment"]["LLAMA_CPP_IDLE_SECONDS"] = "300"
    with patch.object(check_config.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(payload), "")):
        assert check_config.main() == 1
    output = capsys.readouterr()
    assert "secret-do-not-print" not in output.out + output.err
    assert "Lower the idle interval" in output.err

    with patch.object(check_config.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "docker", stderr="secret-do-not-print")):
        assert check_config.main() == 1
    output = capsys.readouterr()
    assert "secret-do-not-print" not in output.out + output.err
