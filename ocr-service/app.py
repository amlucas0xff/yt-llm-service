#!/usr/bin/env python3
"""OCR sidecar service -- accepts images, returns extracted text via PaddleOCR."""

import io
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pynvml
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel


class ImageOCRResult(BaseModel):
    index: int
    texts: list[str]
    confidences: list[float]
    full_text: str
    avg_confidence: float


class OCRBatchResponse(BaseModel):
    results: list[ImageOCRResult]
    engine: str = "paddleocr"
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    engine: str
    gpu: str
    vram_used_mb: float
    vram_total_mb: float


# Global singleton -- initialized on first request
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
    return _ocr_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    pynvml.nvmlInit()
    yield
    pynvml.nvmlShutdown()


app = FastAPI(title="OCR Service", lifespan=lifespan)


@app.post("/ocr", response_model=OCRBatchResponse)
async def ocr_batch(
    images: list[UploadFile] = File(...),
    lang: str = Form("en"),
):
    ocr = get_ocr_engine()
    start = time.perf_counter()
    results = []

    for idx, upload in enumerate(images):
        raw_bytes = await upload.read()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_np = np.array(img)

        texts = []
        confidences = []
        try:
            for result in ocr.predict(img_np):
                rec_texts = result.get("rec_texts", [])
                rec_scores = result.get("rec_scores", [])
                if rec_texts:
                    texts.extend(rec_texts)
                    confidences.extend(rec_scores)
        except Exception:
            pass  # skip failed frame, return empty

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        results.append(ImageOCRResult(
            index=idx,
            texts=texts,
            confidences=confidences,
            full_text="\n".join(texts),
            avg_confidence=avg_conf,
        ))

    elapsed_ms = (time.perf_counter() - start) * 1000
    return OCRBatchResponse(results=results, processing_time_ms=elapsed_ms)


@app.get("/health", response_model=HealthResponse)
async def health():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return HealthResponse(
        status="ok",
        engine="paddleocr",
        gpu=name,
        vram_used_mb=mem.used / (1024 * 1024),
        vram_total_mb=mem.total / (1024 * 1024),
    )
