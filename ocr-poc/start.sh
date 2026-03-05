#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Build if needed
docker compose build

# Create results directory
mkdir -p results

# Run benchmark
docker compose run --rm ocr-poc \
    --frames-dir /app/test_frames \
    --output /app/results/report.json \
    "$@"

echo ""
echo "Results saved to: ocr-poc/results/report.json"
