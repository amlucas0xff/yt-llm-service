# Multi-stage build for YT-Transcriptor
# Stage 1: Base image with CTranslate2 and proper CUDA/cuDNN libraries
FROM ghcr.io/opennmt/ctranslate2:4.6.0-ubuntu22.04-cuda12.2 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables for better performance and reliability
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio/video processing
    ffmpeg \
    # Version control and utilities
    git \
    curl \
    wget \
    # Build dependencies
    build-essential \
    cmake \
    # Audio processing libraries
    libopenblas-dev \
    liblapack-dev \
    libsndfile1 \
    # curl_cffi dependencies
    libcurl4-openssl-dev \
    libssl-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Set up working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/output /app/tmp /app/logs /home/app/.config/matplotlib \
    && chown -R app:app /app /home/app/.config

# Stage 2: Python dependencies
FROM base AS dependencies

# Copy requirements file first for better layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies AS application

# Copy application code and entrypoint script
COPY --chown=app:app . .

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER app

# Set default environment variables (can be overridden)
ENV DEVICE=cuda \
    COMPUTE_TYPE=float16 \
    WHISPER_MODEL=base \
    BATCH_SIZE=16 \
    TEMP_DIR=/app/tmp \
    OUTPUT_DIR=/app/output \
    LOG_LEVEL=INFO \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=0 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    MPLCONFIGDIR=/home/app/.config/matplotlib

# Health check for LLM service
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Expose port for LLM service
EXPOSE 8002

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command is to start the LLM service
CMD ["llm"]
