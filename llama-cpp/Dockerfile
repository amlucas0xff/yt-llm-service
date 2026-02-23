FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Install missing tools: huggingface-cli (for model download) + curl (for healthcheck)
USER root
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends curl python3-pip && \
    pip3 install --no-cache-dir "huggingface_hub[cli]" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
