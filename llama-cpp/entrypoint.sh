#!/bin/bash
set -e

MODEL_DIR="/models"
MODEL_FILE="openai_gpt-oss-20b-MXFP4.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/resolve/main/${MODEL_FILE}"

echo "=== llama-cpp entrypoint ==="
echo "Checking for model at: ${MODEL_PATH}"

# Deliberately no auto-download. Fetching 12GB is a one-time host-side setup
# step, not something a container should do on boot: it turns a missing-file
# mistake into a silent half-hour stall, and it cost the image a whole Python
# toolchain that existed for nothing else.
if [ ! -f "${MODEL_PATH}" ]; then
    cat >&2 <<EOF

ERROR: model not found at ${MODEL_PATH}

The GGUF is not in the image and is not downloaded automatically. Fetch it
once on the host, into ./models/ (about 12GB, no HuggingFace token needed):

    curl -L -o models/${MODEL_FILE} \\
      "${MODEL_URL}"

Then bring the stack up again. ./models is mounted at ${MODEL_DIR}, so the
file only has to be downloaded once per machine, and survives image rebuilds.

EOF
    exit 1
fi

echo "Model found."

GPU_LAYERS="${LLAMA_CPP_GPU_LAYERS:-99}"
echo "Starting llama-server with ${GPU_LAYERS} GPU layers..."

IDLE_SECONDS="${LLAMA_CPP_IDLE_SECONDS:-300}"
echo "Model will unload from VRAM after ${IDLE_SECONDS}s of idle (--sleep-idle-seconds)"

exec /app/llama-server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 131072 \
    --batch-size 2048 \
    --ubatch-size 2048 \
    --n-gpu-layers "${GPU_LAYERS}" \
    --sleep-idle-seconds "${IDLE_SECONDS}" \
    --jinja \
    --log-disable
