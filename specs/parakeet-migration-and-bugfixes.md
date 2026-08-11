# Plan: Parakeet ASR Migration + Codex Bug Fixes

## Task Description

Two coupled work items:

1. **Migrate transcription backend** from WhisperX (`whisper-large-v3-turbo`) to NVIDIA
   Parakeet TDT 0.6B v2 via NeMo. The current model produces word-recognition errors
   ("wrong words") caused by its autoregressive decoder architecture. Parakeet's
   CTC/TDT decoder is non-autoregressive and eliminates the hallucination mechanism at
   the architectural level.

2. **Fix three bugs** identified by Codex review of the Obsidian integration:
   - (HIGH) `user_config.py` crashes the entire service if TOML contains wrong value
     types (e.g. `vault_path = 123`)
   - (MEDIUM) `saved_path` is unbound in three route handlers when disk-save fails,
     causing a silent `UnboundLocalError` that swallows the Obsidian write
   - (LOW) Config parse failures log "disabled (no config or enabled=false)", masking
     the real error from the operator

## Objective

- Replace WhisperX/Whisper with NeMo Parakeet TDT 0.6B v2
- Reduce word error rate from ~7.75% to ~6.05%
- Eliminate hallucination-class transcription errors (non-autoregressive architecture)
- Free ~3-4 GB of VRAM on RTX 4060 (from ~6 GB to ~2-3 GB)
- Fix all three Codex-identified bugs so no config mistake can crash startup or
  silently drop an Obsidian write

## Problem Statement

### Transcription accuracy

`whisper-large-v3-turbo` uses a seq2seq autoregressive decoder. Each output token is
conditioned on all prior output tokens. When audio is ambiguous, the model generates
plausible-sounding text rather than silence ("hallucinations"). WhisperX already applies
the best available mitigations (VAD pre-segmentation, `condition_on_prev_text=False`).
Further improvement requires an architecture change.

Parakeet TDT is a FastConformer encoder + Token-and-Duration Transducer decoder. The
decoder is not autoregressive: all output tokens are produced in a single pass conditioned
only on the encoder output, not on prior predictions. Hallucinations of the Whisper type
are not possible.

### Bugs

```
user_config.py:58   Path(vault_raw)  — crashes if vault_path is non-string TOML value
run_llm_api.py:322  if notes_text and saved_path  — UnboundLocalError when save failed
run_llm_api.py:49   "disabled (no config...)"  — hides parse errors behind misleading msg
```

## Solution Approach

### Phase 1 — Bug fixes (no functional change, safe to ship immediately)

Three targeted edits, no new dependencies.

### Phase 2 — Parakeet migration

Replace `whisperx` with `nemo_toolkit[asr]` in:
- `Dockerfile` (new pip install step)
- `requirements.txt` (add nemo, remove whisperx)
- `src/transcription_service.py` (rewrite `_load_model` and `transcribe_audio`)
- `src/config.py` (rename `WHISPER_MODEL` → `ASR_MODEL`, update default)

The public API (`transcribe_audio` returns same dict shape) must not change so all
callers in `run_llm_api.py` remain untouched.

NeMo Parakeet provides word-level timestamps natively. Speaker diarization is preserved
by keeping pyannote (already a dependency via whisperx). The alignment step (WhisperX
`model_a`) is no longer needed because Parakeet provides timestamps directly.

## Relevant Files

- `src/user_config.py` — bug fix: type-crash on bad TOML value
- `src/run_llm_api.py` — bug fix: `saved_path` unbound + misleading log
- `src/transcription_service.py` — core rewrite: swap WhisperX for NeMo Parakeet
- `src/config.py` — rename `WHISPER_MODEL` → `ASR_MODEL`, update default value
- `requirements.txt` — swap `whisperx` for `nemo_toolkit[asr]`
- `Dockerfile` — add NeMo install; remove whisperx-specific pins

### New Files
None. No new files required.

## Implementation Phases

### Phase 1: Bug Fixes (safe, independent of migration)

Fix all three Codex issues before touching the ASR backend.

### Phase 2: Parakeet Integration

Swap the transcription backend while keeping the output contract identical.

### Phase 3: Validation

Verify service boots, transcribes a short clip, and diarization still works.

## Step by Step Tasks

### 1. Fix HIGH bug — user_config.py type crash

In `src/user_config.py`, wrap the `UserConfig(...)` construction inside the existing
`try/except` block so any `TypeError` from bad TOML values is caught and logged rather
than propagating to startup.

```python
# Before (line ~47-61):
try:
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
except Exception as e:
    logger.warning(f"Failed to parse user config at {config_path}: {e}")
    return None

obsidian = data.get("obsidian", {})
vault_raw = obsidian.get("vault_path", "~/Documents/obsidian")

return UserConfig(
    obsidian_enabled=obsidian.get("enabled", False),
    vault_path=Path(vault_raw).expanduser(),
    ...
)

# After — move UserConfig construction inside the try:
try:
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    obsidian = data.get("obsidian", {})
    vault_raw = str(obsidian.get("vault_path", "~/Documents/obsidian"))
    return UserConfig(
        obsidian_enabled=obsidian.get("enabled", False),
        vault_path=Path(vault_raw).expanduser(),
        inbox_dir=obsidian.get("inbox_dir", "Inbox"),
        tags=obsidian.get("tags", ["video-notes"]),
    )
except Exception as e:
    logger.warning(f"Failed to load user config at {config_path}: {e}")
    return None
```

- Also coerce `vault_raw` with `str(...)` to handle integer TOML values gracefully.

### 2. Fix MEDIUM bug — saved_path potentially unbound in three handlers

In `src/run_llm_api.py`, initialise `saved_path = None` before each `try` block that
assigns it. Affects three independent handlers:

- `transcribe_audio_llm` (~line 285): add `saved_path = None` before the `try`
- `transcribe_youtube_llm` (~line 488): add `saved_path = None` before the `try`
- `transcribe_file_llm` (~line 723): add `saved_path = None` before the `try`

The condition `if notes_text and saved_path:` already handles `None` correctly — the
only change is the initialisation before the try block.

### 3. Fix LOW bug — misleading "disabled" log on parse error

In `src/run_llm_api.py` (~line 46-49), distinguish between "not configured" and "failed
to parse":

```python
# Before:
if obsidian_service:
    logger.info(f"Obsidian integration enabled → {_user_cfg.vault_path / _user_cfg.inbox_dir}")
else:
    logger.info("Obsidian integration disabled (no config or enabled=false)")

# After — check why it's None:
if obsidian_service:
    logger.info(f"Obsidian integration enabled → {_user_cfg.vault_path / _user_cfg.inbox_dir}")
elif _user_cfg is None:
    # load_user_config returns None for both "file missing" and "parse error"
    # The warning from load_user_config already disambiguates; just note the state here.
    logger.info("Obsidian integration disabled (config absent or unreadable — see warnings above)")
else:
    logger.info("Obsidian integration disabled (enabled = false in config)")
```

Note: `load_user_config` already emits a `logger.warning` with the actual error when
parsing fails, so startup logs already carry the root cause at WARNING level. This fix
makes the INFO-level message less misleading.

### 4. Update config.py for Parakeet

In `src/config.py`:
- Rename `WHISPER_MODEL` → `ASR_MODEL`
- Update default from `"large-v3-turbo"` to `"nvidia/parakeet-tdt-0.6b-v2"`
- Remove `COMPUTE_TYPE` (NeMo manages precision internally; no CTranslate2 backend)
- Keep `BATCH_SIZE` and `DEVICE` (NeMo respects these)

```python
# Before:
self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
self.COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")

# After:
self.ASR_MODEL = os.getenv("ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
# COMPUTE_TYPE removed — NeMo handles precision internally
```

Update `__str__` to reference `ASR_MODEL`.

### 5. Rewrite transcription_service.py for NeMo Parakeet

Replace the WhisperX backend while keeping the output contract identical. All callers
receive the same `{"segments": [...], "language": ..., "metadata": {...}}` dict.

```python
def _load_model(self):
    if self.model is None:
        import nemo.collections.asr as nemo_asr
        logger.info(f"Loading Parakeet model: {self.config.ASR_MODEL}")
        self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            self.config.ASR_MODEL
        )
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.cuda()
        logger.info("Parakeet model loaded")

def transcribe_audio(self, audio_path, min_speakers=None, max_speakers=None,
                     batch_size=None, verbose=True):
    audio_path = str(audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    self._load_model()

    # NeMo transcription with word timestamps
    output = self.model.transcribe(
        [audio_path],
        batch_size=batch_size or self.config.BATCH_SIZE,
        return_hypotheses=True,
    )
    hypothesis = output[0][0]  # first file, best hypothesis

    # Build segments compatible with existing format_for_llm interface
    segments = self._hypothesis_to_segments(hypothesis)
    language = "en"  # Parakeet v2 is English-only

    # Optional diarization via pyannote (unchanged)
    if min_speakers or max_speakers:
        segments = self._apply_diarization(audio_path, segments, min_speakers, max_speakers)

    result = {
        "segments": segments,
        "language": language,
        "metadata": {
            "device_used": self.device,
            "model": self.config.ASR_MODEL,
            "batch_size": batch_size or self.config.BATCH_SIZE,
            "speakers_detected": self._count_unique_speakers({"segments": segments}),
            "audio_file": audio_path,
        }
    }
    return result
```

Implement `_hypothesis_to_segments` to convert NeMo's word-level timestamp output
into the `[{"start": float, "end": float, "text": str, "words": [...]}]` format.

Remove `model_a` / `metadata` alignment model attributes — these were WhisperX-specific.

### 6. Update requirements.txt

```
# Remove:
whisperx>=3.1.1

# Add:
nemo_toolkit[asr]>=2.0.0
# Note: pyannote.audio stays (used for diarization)
```

### 7. Update Dockerfile

Add NeMo install after the existing pip install step. NeMo has a complex dependency tree;
pin the install to avoid conflicts with existing torch version:

```dockerfile
# After existing pip install -r requirements.txt:
RUN pip install nemo_toolkit[asr] --no-build-isolation
```

Remove any whisperx-specific workarounds if present.

### 8. Smoke-test in running container

```bash
# Restart service
docker compose restart yt-llm-service

# Wait for healthy
docker compose ps

# Test with a short clip
curl -X POST http://localhost:8002/transcribe-file-llm \
  -F "file=@/tmp/test.mp3" \
  -F "output_format=simple"
```

Confirm:
- No import errors in logs
- `metadata.model` in response is `nvidia/parakeet-tdt-0.6b-v2`
- Transcription text is returned
- VRAM usage is ~2-3 GB (check with `nvidia-smi`)

## Testing Strategy

**Bug fixes** (Phase 1): Can be validated with the existing pytest suite:
```bash
docker compose run --rm --entrypoint="" yt-llm-service \
  python -m pytest tests/test_user_config.py tests/test_obsidian_service.py -v
```
Add one new test to `tests/test_user_config.py`:
```python
def test_load_handles_bad_value_type(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[obsidian]\nenabled = true\nvault_path = 123\n')
    config = load_user_config(config_path=cfg_file)
    assert config is None  # must not raise, must return None
```

**Parakeet migration**: Manual smoke test (no unit tests required — the NeMo model
download is ~1.2 GB and not suitable for unit tests). Acceptance is: service starts,
transcribes a short English clip, returns sensible text.

## Acceptance Criteria

- [ ] `docker compose up yt-llm-service` starts without errors after all changes
- [ ] Startup log shows `Parakeet model loaded` (not WhisperX)
- [ ] `/health` endpoint returns 200
- [ ] `POST /transcribe-file-llm` with a short clip returns transcription text
- [ ] Response `metadata.model` equals `nvidia/parakeet-tdt-0.6b-v2`
- [ ] `nvidia-smi` shows RTX 4060 VRAM usage ≤ 4 GB during transcription
- [ ] `load_user_config` with `vault_path = 123` returns `None` (no crash)
- [ ] `saved_path = None` initialisation present before each of the three `try` blocks
- [ ] Config parse-error log message no longer says "enabled=false" when it was a
      parse failure

## Validation Commands

```bash
# 1. Compile check (all three bug-fixed files)
docker compose run --rm --entrypoint="" yt-llm-service \
  python -m py_compile src/user_config.py src/run_llm_api.py src/config.py src/transcription_service.py

# 2. Unit tests (bug fixes)
docker compose run --rm --entrypoint="" yt-llm-service \
  python -m pytest tests/test_user_config.py -v

# 3. Service health after migration
docker compose up -d yt-llm-service && sleep 10 && curl -s http://localhost:8002/health

# 4. VRAM check during transcription
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

# 5. Full transcription smoke test
curl -X POST http://localhost:8002/transcribe-file-llm \
  -F "file=@/tmp/short_clip.mp3" \
  -F "output_format=simple" | python -m json.tool
```

## Notes

- **Parakeet v2 is English-only.** If multilingual support is needed later, swap to
  `nvidia/parakeet-tdt-0.6b-v3` (25 European languages, WER 6.34% vs 6.05%).
- **NeMo is a large install** (~3 GB). The Docker build will take longer on the first
  run. The model weights (~1.2 GB) are downloaded on first inference and cached at
  `~/.cache/huggingface/hub/` inside the container unless a volume is mounted.
- **Model weight persistence**: Consider mounting a host directory to
  `/root/.cache/huggingface/hub` in docker-compose to avoid re-downloading on container
  rebuild.
- **`COMPUTE_TYPE` env var**: Removing it is a breaking change for any `.env` files
  that set it. Either keep the env var (ignore its value silently) or document the
  removal in the commit message.
- **Bug fixes are independent**: Phase 1 bugs can be fixed and committed before any
  Parakeet work starts. They are safe to merge immediately.
- **WER context**: whisper-large-v3-turbo ≈ 7.75% → Parakeet TDT 0.6B v2 ≈ 6.05%
  on the HuggingFace Open ASR Leaderboard (average across standard benchmarks).
  Real-world improvement on YouTube content may differ.
