# Readiness guardrails — retro items 2–5

**Status:** implemented locally on 2026-10-06. Hosted CI and a real two-GPU run remain unverified. Changes are not committed or pushed.

## Scope and pre-implementation baseline

- Item 2: make the real GPU smoke test safe for existing output and Obsidian notes.
- Item 3: detect when the effective llama-cpp idle setting conflicts with the API's GPU wait limit.
- Item 4: run fast checks in CI.
- Item 5: fix the Obsidian date's time zone and remove obsolete OCR setup from always-loaded agent notes.
- The earlier uncommitted GPU-preflight changes were preserved. Credential rotation from retro item 1 remains separate.
- Before implementation: `make test` passed (58 tests). `scripts/smoke.sh` used the live output directory and overwrote an existing `notes.md` for its default video. There was no CI workflow or active git hook. `shellcheck scripts/smoke.sh` passed. A narrow Ruff check (`E4,E7,E9,F821`) passed; the wider `E4,E7,E9,F` check reported 10 pre-existing errors. The host was UTC−03, but the service container was UTC and lacked tzdata; binding the host's `/etc/localtime` into a temporary container made Python's `date.today()` match the host.

## 1. Safe isolated smoke (item 2)

**Files:** `scripts/smoke.sh`, `Makefile` (only if its target changes), `README.md`; add focused script tests if useful.

1. Keep the short public YouTube URL and the transcript/notes assertions. Replace the live API request with a temporary API container that joins this Compose project's network and talks to the existing llama-cpp sidecar. Discover the Compose network; do not hard-code its generated name. Build the API image only if it is missing.
2. Bind *only* temporary, writable output, temp-audio, config and vault directories to that container, plus the checkout's `src/` read-only so the smoke runs the code under test rather than stale code baked into an image. Enable Obsidian export in the temporary config. Pass the effective service environment without printing `.env` or `docker compose config` (both can contain credentials). Bind the API to a loopback-only, free host port; do not take port 8002 away from the live service.
3. Start llama-cpp if needed, wait for both services' readiness, submit the video with `generate_notes=true`, and assert HTTP 200, non-empty transcript and notes, fresh non-empty `notes.md`, and a vault note with a valid `source` and resolvable `source_transcript`. Use `trap` to remove the temporary container and files on success, error and interruption.
4. Document that `make smoke` is isolated and GPU-bound. Do not add an automatic live-vault write mode. Normal use still goes through the existing API or CLI.

**Verify:** Run the smoke test twice, including once with a pre-existing live note for the default video; compare that file's hash and modification time before and after. Both runs must pass, the live note and live API must be unchanged, and no temporary container or directory may remain. Inject a failing HTTP response once and check cleanup and non-zero exit. `shellcheck scripts/smoke.sh` and `bash -n scripts/smoke.sh` pass. Do not use the present live-writing smoke script to verify this task before isolation is in place.

## 2. Effective GPU configuration check (item 3)

**Files:** `src/notes_service.py`, `docker-compose.yml`, `Makefile`, a small config-check script and its tests, `README.md`.

1. Name the API's current 30-second wait limit rather than leaving it as an unexplained loop count. Keep the existing 5-second same-GPU idle default. Define the policy: when both services share a GPU, the *effective* idle interval must leave time inside that wait limit; `-1` is invalid for that shared-GPU setup. Do not try to unload arbitrary `nvidia-smi` PIDs.
2. Add `make check-config` to inspect **only** the relevant resolved Compose settings. Do not print the full Compose config, environment or tokens. Fail with an actionable message if the sidecar would not sleep before the API gives up. Account for `.env` overrides, not just checked-in defaults. For the two-GPU layout already documented in README, make an explicit Compose override mark the GPUs as separate; the API must then skip this wait and the check must not require idle unload. Do not guess the GPU layout from process names.
3. Keep transcription available when the optional sidecar is unreachable, as it is now; distinguish that case from a reachable sidecar that remains awake too long.

**Verify:** With no override and with `LLAMA_CPP_IDLE_SECONDS=5`, the check passes. With `300` or `-1` on a shared GPU, it fails *before* a transcription request and prints no secrets. With an explicit two-GPU override and idle disabled, the API does not wait for llama-cpp. Unit tests cover these cases; the real sidecar and API stay healthy with the default. The isolated smoke from step 1 passes while the sidecar was recently active.

## 3. Enforce fast checks in CI (item 4)

**Files:** `.github/workflows/checks.yml`, `Makefile` only if a shared check target helps, `README.md`.

1. Add one GitHub Actions job on pushes and pull requests: install the project's `uv`/Python toolchain, run `make test`, `docker compose config --quiet`, `make check-config` with checkout defaults, `bash -n` and ShellCheck on `scripts/smoke.sh`.
2. Add the Ruff rules already clean on this checkout: `E4,E7,E9,F821` for `src/`, `tests/` and `cli.py`. Do **not** turn on all `F` rules in this task: the current baseline has 10 failures and includes imports used as dependency probes. Record that wider lint needs a separate, reviewed cleanup.
3. Keep the real GPU smoke test manual; hosted CI has neither the model nor the NVIDIA GPU. Do not require `.env`, a Hugging Face token or a 12 GB model download for the fast job.

**Verify:** Run each CI command locally from a clean checkout environment, then confirm the workflow passes on a push or PR. A deliberate failing unit test and a syntax error in a copy of the shell script must make the matching check fail. CI logs must not disclose local config values.

## 4. Local date and agent-note cleanup (item 5)

**Files:** `docker-compose.yml`, `tests/test_obsidian_service.py` or a small container check, `README.md`, `CLAUDE.md`.

1. On this Linux deployment, bind `/etc/localtime` read-only into `yt-llm-service`. Do not rely on `TZ=America/Sao_Paulo` alone: this image lacks tzdata. Document the Linux-host assumption. Keep `ObsidianService` using the container's local `date.today()`; no new date-setting API is needed.
2. Add a check for the Compose mount and keep the existing frontmatter date test. After recreation, compare the host and container UTC offsets and dates. Have the **isolated** smoke test check its new vault note's date against the host date *before* it removes the temporary vault. Do not automatically rewrite existing vault notes with the old UTC date.
3. Prune the removed OCR proof-of-concept installation sections from `CLAUDE.md`. Keep only active test/smoke navigation and the import-time test setup warning. The historical OCR decision remains in `docs/adr/0002-abandonar-ocr-de-video.md`; link there only if a history pointer is needed. Update any smoke guidance to describe the isolated command.

**Verify:** The container reports the host's offset (currently `-0300`), a newly generated note has the host-local date, `make test` and `make smoke` pass, and existing vault notes remain unchanged. `CLAUDE.md` has no obsolete OCR setup recipe.

## Order and stop conditions

Implement steps **1 → 2 → 3 → 4**. Re-run `make test`, `make check-config` and `docker compose config --quiet` after each relevant step; run the GPU smoke test only after step 1 has made it safe. Stop and ask before any change that would require rewriting user notes, terminating unrelated GPU processes, or printing credential values. Leave unrelated code and the 10 existing wider-Ruff findings for separate work.

## Local verification (2026-10-06)

- `make test`: 91 passed. `make check-config`, `docker compose config --quiet`, `bash -n scripts/smoke.sh`, ShellCheck, and Ruff `E4,E7,E9,F821` passed.
- Real isolated `make smoke` passed after integration. The existing default-video note's hash and modification time did not change; no smoke container or scratch directory remained. The new temporary note had the host-local date and a resolvable transcript link.
- Negative GPU-config and split-GPU routing cases have unit tests. This host has one GPU, so the two-GPU runtime path was not exercised on hardware.
- The live API was restarted after code changes and is healthy with the same host-local date as the host. No GitHub-hosted CI run occurred because no commit or push was requested.
