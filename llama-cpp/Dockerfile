FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Install missing tools: huggingface-cli (for model download) + curl (for healthcheck)
USER root
# --break-system-packages: the base image's Python is PEP 668 externally-managed,
# which makes a bare `pip3 install` refuse outright. Inside a single-purpose
# container there is no system Python to protect, so installing into it is fine.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends curl python3-pip && \
    pip3 install --no-cache-dir --break-system-packages "huggingface_hub[cli]" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
