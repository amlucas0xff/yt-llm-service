#!/usr/bin/env bash
#
# End-to-end smoke test: the only check that exercises WhisperX and llama-cpp
# for real. `make test` mocks both, so a green unit suite says nothing about
# whether transcription and notes generation actually work.
#
# Slow (minutes) and GPU-bound — run on demand, not on every commit.
#
# Usage:
#   ./scripts/smoke.sh [youtube-url]
#
# Environment:
#   HOST_OUTPUT_DIR   where the service writes results (default: ./data/output)
#   SMOKE_TIMEOUT     seconds to wait for the transcription response (default: 900)

set -euo pipefail

# "Me at the zoo" — 19s, has speech, and as the first video ever uploaded to
# YouTube it is about as unlikely to disappear as a URL gets.
URL="${1:-https://www.youtube.com/watch?v=jNQXAC9IVRw}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# HOST_OUTPUT_DIR normally lives in .env, which docker compose reads but the
# shell does not — so look there before falling back, or the note check hunts
# in ./data/output while the service writes somewhere else entirely.
if [ -z "${HOST_OUTPUT_DIR:-}" ] && [ -f .env ]; then
    HOST_OUTPUT_DIR=$(grep -E '^HOST_OUTPUT_DIR=' .env | tail -n 1 | cut -d= -f2- | tr -d "\"'")
fi
OUTPUT_DIR="${HOST_OUTPUT_DIR:-./data/output}"
TIMEOUT="${SMOKE_TIMEOUT:-900}"
BASE_URL="${SMOKE_BASE_URL:-http://localhost:8002}"

RESPONSE_FILE="$(mktemp)"
trap 'rm -f "$RESPONSE_FILE"' EXIT

fail() {
    echo "SMOKE FAIL: $*" >&2
    exit 1
}

echo "==> Starting services"
docker compose up -d --wait || fail "docker compose up did not come up healthy"

echo "==> Transcribing $URL (timeout ${TIMEOUT}s)"
started=$(date +%s)
http_code=$(
    curl -sS --max-time "$TIMEOUT" \
        -o "$RESPONSE_FILE" -w '%{http_code}' \
        -X POST "$BASE_URL/transcribe-youtube-llm" \
        -H 'Content-Type: application/json' \
        -d "{\"youtube_url\": \"$URL\", \"output_format\": \"simple\", \"generate_notes\": true}"
) || fail "request to $BASE_URL/transcribe-youtube-llm failed"
elapsed=$(( $(date +%s) - started ))

echo "==> HTTP $http_code in ${elapsed}s"
[ "$http_code" = "200" ] || fail "expected HTTP 200, got $http_code: $(head -c 500 "$RESPONSE_FILE")"

python3 - "$RESPONSE_FILE" <<'PY' || fail "response body did not carry a transcript"
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

if not data.get("success"):
    print(f"success was {data.get('success')!r}: {data.get('error')!r}", file=sys.stderr)
    sys.exit(1)

text = (data.get("text") or "").strip()
if not text:
    print("transcript text was empty", file=sys.stderr)
    sys.exit(1)

print(f"    transcript: {len(text)} chars — {text[:80]!r}")

notes = (data.get("notes") or "").strip()
if not notes:
    print("notes were empty (llama-cpp did not generate)", file=sys.stderr)
    sys.exit(1)

print(f"    notes: {len(notes)} chars")
PY

echo "==> Checking $OUTPUT_DIR for a written note"
[ -d "$OUTPUT_DIR" ] || fail "output dir $OUTPUT_DIR does not exist"

# The service writes to {OUTPUT_DIR}/{sanitized video title}/notes.md. Match on
# "written since this run started" rather than guessing at the sanitized title.
# find returns directory order, not newest-first, so take any match — the
# -newermt filter is what makes it this run's note.
note=$(find "$OUTPUT_DIR" -name notes.md -newermt "@$started" -print 2>/dev/null | head -n 1)
[ -n "$note" ] || fail "no notes.md written under $OUTPUT_DIR since this run started"
[ -s "$note" ] || fail "note $note is empty"

echo "    note: $note ($(wc -c < "$note") bytes)"
echo
echo "SMOKE PASS in ${elapsed}s"
