#!/usr/bin/env python3
"""Validate GPU wait policy against *resolved* Compose settings, without logging config."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpu_policy import GPU_WAIT_SECONDS


class ConfigError(Exception):
    """Safe, credential-free diagnostic for the operator."""


def pinned_gpu_ids(service: dict) -> list[str]:
    """Require one unambiguous, Docker-safe index reservation."""
    try:
        devices = service["deploy"]["resources"]["reservations"]["devices"]
        if len(devices) != 1 or "count" in devices[0]:
            raise ValueError("ambiguous reservation")
        reservation = devices[0]
        ids = reservation["device_ids"]
        if (reservation.get("driver") != "nvidia"
                or "gpu" not in reservation.get("capabilities", [])
                or not isinstance(ids, list) or not ids
                or any(not isinstance(i, str) or not i.isascii() or not i.isdecimal() for i in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("invalid GPU IDs")
        return ids
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("GPU reservation requires one explicit NVIDIA device_ids list of GPU indices") from exc


def smoke_gpu_device(config: dict) -> str:
    """GPU argument for the isolated API from its resolved Compose config."""
    check(config)
    api = config["services"]["yt-llm-service"]
    sidecar = config["services"]["llama-cpp"]
    api_ids = pinned_gpu_ids(api)
    if api["environment"]["SIDECAR_GPU_SEPARATE"] == "false":
        if set(api_ids) != set(pinned_gpu_ids(sidecar)):
            raise ConfigError("Smoke shared GPU reservations must match; pin both services or use a split override")
    device = "device=" + ",".join(api_ids)
    # Docker CLI parses --gpus as CSV; multiple IDs require inner CSV quotes.
    return f'"{device}"' if len(api_ids) > 1 else device


def check(config: dict) -> str:
    try:
        services = config["services"]
        api = services["yt-llm-service"]
        sidecar = services["llama-cpp"]
        marker = api["environment"]["SIDECAR_GPU_SEPARATE"]
        idle = int(sidecar["environment"]["LLAMA_CPP_IDLE_SECONDS"])
        api_idle = int(api["environment"]["LLAMA_CPP_IDLE_SECONDS"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("Missing or invalid GPU policy settings in resolved Compose services") from exc

    if api_idle != idle:
        raise ConfigError("API and llama-cpp must use the same LLAMA_CPP_IDLE_SECONDS")
    if marker not in ("true", "false"):
        raise ConfigError("SIDECAR_GPU_SEPARATE must be exactly true or false in the API service")
    if idle < -1:
        raise ConfigError("LLAMA_CPP_IDLE_SECONDS must be -1 or a nonnegative integer")

    if marker == "true":
        # The marker is a promise: a count reservation cannot prove separation.
        api_ids, sidecar_ids = set(pinned_gpu_ids(api)), set(pinned_gpu_ids(sidecar))
        if api_ids & sidecar_ids:
            raise ConfigError("Split-GPU mode requires disjoint GPU device_ids for API and llama-cpp")
        return "GPU config OK: explicitly pinned to separate GPUs; API skips sidecar wait"

    if idle == -1 or idle >= GPU_WAIT_SECONDS:
        raise ConfigError(
            f"Shared GPU: LLAMA_CPP_IDLE_SECONDS must be 0..{GPU_WAIT_SECONDS - 1} "
            f"so llama-cpp sleeps before the API's {GPU_WAIT_SECONDS}s wait ends. "
            "Lower the idle interval, or explicitly pin separate GPUs and set SIDECAR_GPU_SEPARATE=true."
        )
    return f"GPU config OK: shared GPU; idle unload precedes the API's {GPU_WAIT_SECONDS}s wait"


def main() -> int:
    try:
        # Never forward stdout/stderr: Compose JSON contains the entire environment,
        # including credentials. Even malformed JSON and subprocess errors stay private.
        result = subprocess.run(
            ["docker", "compose", "config", "--format", "json"],
            capture_output=True, check=True, text=True,
        )
        message = check(json.loads(result.stdout))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        print("GPU config check failed: could not resolve Compose config (run docker compose config --quiet)", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"GPU config check failed: {exc}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
