FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps for PaddleOCR (libGL, libglib for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PaddlePaddle GPU (CUDA 12.3 wheels) without NVIDIA runtime deps
# since the base image already provides CUDA 12.4 runtime (forward-compatible).
# Then install its missing Python-only deps (protobuf etc.) separately.
RUN pip install --no-cache-dir --no-deps \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu123/ \
    paddlepaddle-gpu==3.1.0 && \
    pip install --no-cache-dir protobuf numpy decorator opt-einsum

# Install PaddleOCR and its non-NVIDIA deps
RUN pip install --no-cache-dir paddleocr>=2.7.0

# Install Surya OCR v0.6.x (compatible with torch 2.5) without pulling
# a conflicting torch version via --no-deps, then install its Python deps
RUN pip install --no-cache-dir --no-deps surya-ocr==0.6.13 && \
    pip install --no-cache-dir \
    pdftext filetype pydantic "transformers>=4.36.0,<4.46" huggingface_hub tokenizers \
    einops opencv-python-headless ftfy

# Remaining benchmark deps
RUN pip install --no-cache-dir Pillow>=10.0.0 pynvml>=12.0.0

COPY benchmark.py .

ENTRYPOINT ["python", "benchmark.py"]
