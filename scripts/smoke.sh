#!/usr/bin/env bash
# Real WhisperX + llama-cpp smoke against a disposable API and scratch vault.
# Never mount the live output directory, config, or Obsidian vault.
set -euo pipefail

URL="${1:-https://www.youtube.com/watch?v=jNQXAC9IVRw}"
TIMEOUT="${SMOKE_TIMEOUT:-900}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }
command -v docker >/dev/null || fail "docker is required"
command -v python3 >/dev/null || fail "python3 is required"
command -v curl >/dev/null || fail "curl is required"

scratch="$(mktemp -d -t yt-llm-smoke.XXXXXXXX)"
container="smoke-api-$(basename "$scratch" | tr -cd 'a-zA-Z0-9')"
cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
    rm -rf -- "$scratch"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$scratch"/{output,tmp,logs,uploads,config,vault}
# The image runs as app (UID 1000 on this host). Use the invoking user's UID
# for scratch ownership; don't grant the container access to other host paths.
cat > "$scratch/config/config.toml" <<EOF
[obsidian]
enabled = true
vault_path = "$scratch/vault"
inbox_dir = "Inbox"
EOF

# Resolve the *service* environment, not the .env file itself: Compose also
# supplies defaults and may use an override. Never print the resolved config.
# Write a Docker env file directly; its permissions and lifetime are limited.
(umask 077; docker compose config --format json | python3 -c '
import json, sys
sys.path.insert(0, "scripts")
from check_config import ConfigError, pinned_gpu_ids, smoke_gpu_device
config = json.load(sys.stdin)
try:
    gpu = smoke_gpu_device(config)
except ConfigError as exc:
    raise SystemExit(f"Unsafe smoke GPU configuration: {exc}")
with open(sys.argv[2], "w", encoding="utf-8") as out:
    out.write(gpu)
with open(sys.argv[3], "w", encoding="utf-8") as out:
    out.write(",".join(pinned_gpu_ids(config["services"]["llama-cpp"])))
service = config["services"]["yt-llm-service"]
with open(sys.argv[1], "w", encoding="utf-8") as out:
    for key, value in service.get("environment", {}).items():
        if key in {"HOST_OUTPUT_DIR", "TEMP_DIR", "OUTPUT_DIR"}:
            continue
        value = "" if value is None else str(value)
        if "\n" in value or "\r" in value:
            raise ValueError("multiline service environment is unsupported")
        out.write(f"{key}={value}\n")
' "$scratch/service.env" "$scratch/gpu" "$scratch/sidecar-gpu") || fail "could not resolve safe service environment and GPU reservation"

# Only start the sidecar if absent/stopped; don't recreate or restart the live
# API. The sidecar's attached Compose network is discovered, not guessed.
sidecar="$(docker compose ps -q llama-cpp)" || fail "could not find sidecar"
if [ -z "$sidecar" ] || [ "$(docker inspect -f '{{.State.Running}}' "$sidecar")" != true ]; then
    echo "==> Starting llama-cpp sidecar"
    docker compose up -d --no-deps llama-cpp >/dev/null || fail "could not start llama-cpp"
    sidecar="$(docker compose ps -q llama-cpp)"
fi
[ -n "$sidecar" ] || fail "llama-cpp is not available"
network="$(docker inspect "$sidecar" | SIDECAR_GPU_FILE="$scratch/sidecar-gpu" python3 -c '
import json, os, subprocess, sys
container = json.load(sys.stdin)[0]
expected = set(open(os.environ["SIDECAR_GPU_FILE"], encoding="utf-8").read().split(","))
requests = container["HostConfig"].get("DeviceRequests") or []
if (len(requests) != 1 or requests[0].get("Driver") != "nvidia"
        or set(requests[0].get("DeviceIDs") or []) != expected):
    raise SystemExit("Running sidecar GPU reservation differs from resolved Compose config; recreate the sidecar first")
project = container["Config"]["Labels"].get("com.docker.compose.project")
for name in container["NetworkSettings"]["Networks"]:
    net = json.loads(subprocess.check_output(["docker", "network", "inspect", name]))[0]
    labels = net.get("Labels") or {}
    if labels.get("com.docker.compose.project") == project and labels.get("com.docker.compose.network") == "default":
        print(name)
        break
' )" || fail "could not inspect sidecar network"
[ -n "$network" ] || fail "sidecar is not attached to the default Compose network"

echo "==> Waiting for llama-cpp health"
for _ in $(seq 1 180); do
    [ "$(docker inspect -f '{{.State.Health.Status}}' "$sidecar" 2>/dev/null || true)" = healthy ] && break
    sleep 2
done
[ "$(docker inspect -f '{{.State.Health.Status}}' "$sidecar")" = healthy ] || fail "llama-cpp did not become healthy"

# Build only when no local API image exists. The read-only src mount ensures
# the temporary API tests this checkout even when the image is older.
image="$(docker compose config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["yt-llm-service"]["image"])')" || fail "could not resolve API image"
if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "==> Building missing API image"
    docker compose build yt-llm-service >/dev/null || fail "API image build failed"
fi

echo "==> Starting isolated API"
gpu="$(<"$scratch/gpu")"
docker run -d --rm --name "$container" --network "$network" --gpus "$gpu" \
    --user "$(id -u):$(id -g)" --env-file "$scratch/service.env" \
    -e HOME=/home/app -e HOST_OUTPUT_DIR="$scratch/output" \
    -e TEMP_DIR=/app/tmp -e OUTPUT_DIR=/app/output \
    -p 127.0.0.1::8002 \
    --mount "type=bind,src=/etc/localtime,dst=/etc/localtime,readonly" \
    --mount "type=bind,src=$REPO_ROOT/src,dst=/app/src,readonly" \
    --mount "type=bind,src=$scratch/output,dst=/app/output" \
    --mount "type=bind,src=$scratch/tmp,dst=/app/tmp" \
    --mount "type=bind,src=$scratch/logs,dst=/app/logs" \
    --mount "type=bind,src=$scratch/uploads,dst=/app/uploads" \
    --mount "type=bind,src=$scratch/config,dst=/home/app/.config/yt-llm,readonly" \
    --mount "type=bind,src=$scratch/vault,dst=$scratch/vault" \
    "$image" >/dev/null || fail "could not start isolated API"
port="$(docker port "$container" 8002/tcp | awk -F: '$1 == "127.0.0.1" {print $NF}')"
[ -n "$port" ] || fail "could not resolve isolated API loopback port"
BASE_URL="http://127.0.0.1:$port"
for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 "$BASE_URL/health" -o /dev/null 2>/dev/null; then break; fi
    [ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" = true ] || fail "isolated API exited before health check"
    sleep 2
done
curl -fsS --max-time 2 "$BASE_URL/health" -o /dev/null || fail "isolated API did not become healthy"

# Test-only failure hook: exercise cleanup without making a transcription request.
if [ "${SMOKE_TEST_FAIL_AFTER_HEALTH:-0}" = 1 ]; then
    fail "injected failure after isolated API health"
fi
echo "==> Transcribing $URL (timeout ${TIMEOUT}s)"
started="$(date +%s)"
start_date="$(date +%F)"
python3 - "$URL" "$scratch/request.json" <<'PY'
import json, sys
with open(sys.argv[2], 'w') as f:
    json.dump({'youtube_url': sys.argv[1], 'output_format': 'simple', 'generate_notes': True}, f)
PY
http_code="$(curl -sS --max-time "$TIMEOUT" -o "$scratch/response.json" -w '%{http_code}' \
    -X POST "$BASE_URL/transcribe-youtube-llm" -H 'Content-Type: application/json' \
    --data-binary "@$scratch/request.json")" || fail "transcription request failed"
echo "==> HTTP $http_code in $(( $(date +%s) - started ))s"
[ "$http_code" = 200 ] || fail "expected HTTP 200, got $http_code"

python3 - "$scratch/response.json" "$scratch/output" "$scratch/vault" "$URL" "$started" "$start_date" "$(date +%F)" <<'PY' || fail "response or persisted notes/provenance invalid"
import json
import pathlib
import sys
from urllib.parse import parse_qs, urlparse

response, output, vault = map(pathlib.Path, sys.argv[1:4])
url, started, start_date, end_date = sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7]
data = json.loads(response.read_text())
assert data.get('success') is True, 'transcription failed'
assert (data.get('text') or '').strip(), 'empty transcript'
assert (data.get('notes') or '').strip(), 'empty notes'
notes = [p for p in output.rglob('notes.md') if p.stat().st_mtime >= started and p.stat().st_size]
assert len(notes) == 1, f'expected one fresh non-empty notes.md, got {len(notes)}'
vault_notes = list(vault.rglob('*.md'))
assert len(vault_notes) == 1, f'expected one vault note, got {len(vault_notes)}'
text = vault_notes[0].read_text()
assert text.startswith('---\n') and '\n---\n' in text[4:], 'missing frontmatter'
fields = {}
for line in text.split('---', 2)[1].splitlines():
    if ': ' in line:
        key, value = line.split(': ', 1)
        fields[key] = value
assert fields.get('date') in (start_date, end_date), 'vault date differs from host-local date'
source = fields.get('source', '')
parsed = urlparse(source)
assert parsed.scheme in ('http', 'https') and parsed.netloc in ('youtube.com', 'www.youtube.com', 'youtu.be'), 'invalid source'
expected = urlparse(url)
if expected.netloc in ('youtube.com', 'www.youtube.com'):
    assert parse_qs(parsed.query).get('v') == parse_qs(expected.query).get('v'), 'wrong video source'
else:
    assert source == url, 'wrong video source'
transcript = fields.get('source_transcript', '')
assert transcript.startswith("'") and transcript.endswith("'"), 'missing quoted source_transcript'
path = pathlib.Path(transcript[1:-1].replace("''", "'"))
assert path.resolve().is_relative_to(output.resolve()) and path.is_file() and path.stat().st_size, 'source_transcript does not resolve in scratch output'
print(f'    transcript: {len(data["text"])} chars; notes: {notes[0].stat().st_size} bytes; vault provenance OK')
PY

echo "SMOKE PASS"
