"""Async HTTP client for the OCR sidecar service."""

import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OCRFrameResult:
    index: int
    texts: list[str]
    confidences: list[float]
    full_text: str
    avg_confidence: float


@dataclass
class OCRBatchResult:
    results: list[OCRFrameResult]
    engine: str
    processing_time_ms: float


class OCRClient:
    """Async client for the ocr-service sidecar."""

    def __init__(self, base_url: str = "http://ocr-service:8003", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0)

    async def ocr_images(
        self,
        image_paths: list[str],
        lang: str = "en",
    ) -> OCRBatchResult | None:
        """Send images to OCR service and return results.

        Returns None if the service is unavailable or errors out.
        """
        if not image_paths:
            return OCRBatchResult(results=[], engine="paddleocr", processing_time_ms=0.0)

        files = []
        for path in image_paths:
            p = Path(path)
            files.append(("images", (p.name, open(p, "rb"), "image/png")))

        url = f"{self.base_url}/ocr"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, files=files, data={"lang": lang})
                response.raise_for_status()

            data = response.json()
            results = [
                OCRFrameResult(
                    index=r["index"],
                    texts=r["texts"],
                    confidences=r["confidences"],
                    full_text=r["full_text"],
                    avg_confidence=r["avg_confidence"],
                )
                for r in data.get("results", [])
            ]
            return OCRBatchResult(
                results=results,
                engine=data.get("engine", "paddleocr"),
                processing_time_ms=data.get("processing_time_ms", 0.0),
            )

        except httpx.ConnectError as e:
            logger.warning(f"ocr-service unavailable: {e}")
            return None
        except httpx.TimeoutException as e:
            logger.warning(f"ocr-service request timed out: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"ocr-service HTTP error {e.response.status_code}: {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling ocr-service: {e}")
            return None
        finally:
            for _, (_, fobj, _) in files:
                fobj.close()

    async def health(self) -> dict | None:
        """Check ocr-service health. Returns health dict or None."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return None
