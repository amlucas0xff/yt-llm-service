FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PaddlePaddle GPU with --no-deps to avoid CUDA runtime conflicts
RUN pip install --no-cache-dir --no-deps \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu123/ \
    paddlepaddle-gpu==3.1.0 && \
    pip install --no-cache-dir protobuf numpy decorator opt-einsum

# PaddleOCR + remaining deps
RUN pip install --no-cache-dir paddleocr>=2.7.0

# FastAPI + web deps
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.23.0 \
    pynvml>=12.0.0 \
    python-multipart>=0.0.6 \
    Pillow>=10.0.0

COPY app.py .

EXPOSE 8003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
