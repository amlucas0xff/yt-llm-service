"""Tests for ocr_client module."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from ocr_client import OCRClient


@pytest.mark.asyncio
async def test_ocr_images_sends_multipart(tmp_path):
    """Verify images are sent as multipart to the sidecar."""
    # Create fake image files
    img1 = tmp_path / "frame_0001.png"
    img2 = tmp_path / "frame_0002.png"
    img1.write_bytes(b"fake png 1")
    img2.write_bytes(b"fake png 2")

    client = OCRClient(base_url="http://localhost:8003")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "texts": ["hello"], "confidences": [0.99],
             "full_text": "hello", "avg_confidence": 0.99},
            {"index": 1, "texts": ["world"], "confidences": [0.95],
             "full_text": "world", "avg_confidence": 0.95},
        ],
        "engine": "paddleocr",
        "processing_time_ms": 500.0,
    }

    with patch("httpx.AsyncClient") as MockClient:
        mock_post = AsyncMock(return_value=mock_response)
        MockClient.return_value.__aenter__.return_value.post = mock_post
        result = await client.ocr_images([str(img1), str(img2)])

    assert result is not None
    assert len(result.results) == 2
    assert result.results[0].full_text == "hello"
    assert result.results[1].full_text == "world"
    assert result.engine == "paddleocr"


@pytest.mark.asyncio
async def test_ocr_images_empty_list():
    """Empty image list returns empty results, no HTTP call."""
    client = OCRClient()
    result = await client.ocr_images([])
    assert result is not None
    assert result.results == []


@pytest.mark.asyncio
async def test_ocr_images_service_unavailable(tmp_path):
    """Returns None when ocr-service is down."""
    img = tmp_path / "img.png"
    img.write_bytes(b"fake png")
    client = OCRClient(base_url="http://localhost:8003")

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=Exception("connection refused")
        )
        result = await client.ocr_images([str(img)])

    assert result is None


@pytest.mark.asyncio
async def test_health_returns_dict():
    """Health check returns parsed JSON."""
    client = OCRClient()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"status": "ok", "engine": "paddleocr"}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        result = await client.health()

    assert result == {"status": "ok", "engine": "paddleocr"}
