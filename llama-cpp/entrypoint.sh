#!/bin/bash
set -e

MODEL_DIR="/models"
MODEL_FILE="openai_gpt-oss-20b-MXFP4.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
REPO="bartowski/openai_gpt-oss-20b-GGUF"

echo "=== llama-cpp entrypoint ==="
echo "Checking for model at: ${MODEL_PATH}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "Model not found. Downloading ${MODEL_FILE} from HuggingFace (~12GB)..."
    echo "This may take several minutes on first run."
    huggingface-cli download "${REPO}" \
        --include "${MODEL_FILE}" \
        --local-dir "${MODEL_DIR}"
    echo "Download complete."
else
    echo "Model found. Skipping download."
fi

GPU_LAYERS="${LLAMA_CPP_GPU_LAYERS:-99}"
echo "Starting llama-server with ${GPU_LAYERS} GPU layers..."

exec llama-server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 131072 \
    --batch-size 2048 \
    --ubatch-size 2048 \
    --n-gpu-layers "${GPU_LAYERS}" \
    --jinja \
    --log-disable
