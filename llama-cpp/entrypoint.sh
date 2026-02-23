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
    python3 -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id='${REPO}',
    filename='${MODEL_FILE}',
    local_dir='${MODEL_DIR}',
    token=os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN'),
)
print('Download complete.')
"
else
    echo "Model found. Skipping download."
fi

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
