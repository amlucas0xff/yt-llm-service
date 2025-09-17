"""
FastAPI application for the LLM Transcription Service
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import os
import logging
from pathlib import Path
import tempfile
import shutil

from config import Config
from transcription_service import TranscriptionService
from audio_downloader import AudioDownloader
from simple_logger import log_action

# Initialize configuration first
config = Config()

# Configure logging using config value
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize services
transcription_service = TranscriptionService(config)
audio_downloader = AudioDownloader(temp_dir=config.TEMP_DIR)

# Create FastAPI app
app = FastAPI(
    title="YT-LLM Transcription Service",
    description="Dedicated service for audio transcription using WhisperX",
    version="1.0.0",
)


class TranscriptionRequest(BaseModel):
    """Request model for transcription"""

    audio_file_path: str
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    batch_size: Optional[int] = None
    verbose: bool = True


class YouTubeTranscriptionRequest(BaseModel):
    """Request model for YouTube transcription"""

    youtube_url: str
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    batch_size: Optional[int] = None
    verbose: bool = True


class MetadataExtractionRequest(BaseModel):
    """Request model for metadata extraction"""

    youtube_url: str


class TranscriptionResponse(BaseModel):
    """Response model for transcription"""

    success: bool
    segments: list
    language: Optional[str] = None
    metadata: dict
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    service: str
    device_info: dict


class LLMTranscriptionRequest(BaseModel):
    """Request model for LLM-optimized transcription"""

    audio_file_path: str
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    batch_size: Optional[int] = None
    output_format: str = "simple"  # simple, speaker, structured, markdown
    remove_filler_words: bool = False
    merge_consecutive_speakers: bool = True
    verbose: bool = True


class YouTubeLLMTranscriptionRequest(BaseModel):
    """Request model for YouTube LLM-optimized transcription"""

    youtube_url: str
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    batch_size: Optional[int] = None
    output_format: str = "simple"  # simple, speaker, structured, markdown
    remove_filler_words: bool = False
    merge_consecutive_speakers: bool = True
    verbose: bool = True


class LLMTranscriptionResponse(BaseModel):
    """Response model for LLM-optimized transcription"""

    success: bool
    text: Optional[str] = None
    speakers: Optional[dict] = None
    blocks: Optional[list] = None
    language: Optional[str] = None
    metadata: dict
    error: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        service="yt-llm-transcription",
        device_info=transcription_service.get_device_info(),
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcribe audio file with optional speaker diarization
    """
    log_action(
        f"Received transcription request for audio file: {request.audio_file_path}"
    )
    logger.info(f"Received transcription request for: {request.audio_file_path}")
    logger.debug(
        f"Request parameters: min_speakers={request.min_speakers}, max_speakers={request.max_speakers}, batch_size={request.batch_size}, verbose={request.verbose}"
    )

    try:
        # Validate audio file exists
        audio_path = Path(request.audio_file_path)
        logger.debug(f"Checking if audio file exists: {audio_path}")
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Audio path absolute: {audio_path.absolute()}")

        if not audio_path.exists():
            logger.error(f"Audio file not found: {request.audio_file_path}")
            logger.debug(
                f"Files in parent directory: {list(audio_path.parent.iterdir()) if audio_path.parent.exists() else 'Parent does not exist'}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Audio file not found: {request.audio_file_path}",
            )

        file_size = audio_path.stat().st_size
        logger.info(
            f"Audio file found: {request.audio_file_path} ({file_size} bytes, {file_size / 1024 / 1024:.2f} MB)"
        )

        # Perform transcription
        result = transcription_service.transcribe_audio(
            audio_path=request.audio_file_path,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            batch_size=request.batch_size,
            verbose=request.verbose,
        )

        return TranscriptionResponse(
            success=True,
            segments=result.get("segments", []),
            language=result.get("language"),
            metadata=result.get("metadata", {}),
            error=None,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/transcribe-llm", response_model=LLMTranscriptionResponse)
async def transcribe_audio_llm(request: LLMTranscriptionRequest):
    """
    Transcribe audio file and format for LLM consumption
    """
    log_action(
        f"Received LLM transcription request for audio file: {request.audio_file_path}"
    )
    logger.info(f"Received LLM transcription request for: {request.audio_file_path}")
    logger.debug(
        f"Request parameters: min_speakers={request.min_speakers}, max_speakers={request.max_speakers}, "
        f"batch_size={request.batch_size}, output_format={request.output_format}, "
        f"remove_filler_words={request.remove_filler_words}, verbose={request.verbose}"
    )

    try:
        # Validate audio file exists
        audio_path = Path(request.audio_file_path)
        logger.debug(f"Checking if audio file exists: {audio_path}")
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Audio path absolute: {audio_path.absolute()}")

        if not audio_path.exists():
            logger.error(f"Audio file not found: {request.audio_file_path}")
            logger.debug(
                f"Files in parent directory: {list(audio_path.parent.iterdir()) if audio_path.parent.exists() else 'Parent does not exist'}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Audio file not found: {request.audio_file_path}",
            )

        file_size = audio_path.stat().st_size
        logger.info(
            f"Audio file found: {request.audio_file_path} ({file_size} bytes, {file_size / 1024 / 1024:.2f} MB)"
        )

        # Perform transcription
        result = transcription_service.transcribe_audio(
            audio_path=request.audio_file_path,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            batch_size=request.batch_size,
            verbose=request.verbose,
        )

        # Format for LLM consumption
        llm_result = transcription_service.format_for_llm(
            transcription_result=result,
            output_format=request.output_format,
            include_speakers=(request.output_format in ["speaker", "structured", "markdown"]),
            merge_consecutive_speakers=request.merge_consecutive_speakers,
            remove_filler_words=request.remove_filler_words
        )

        # Build response based on format
        response_data = {
            "success": True,
            "language": result.get("language"),
            "metadata": llm_result.get("metadata", {}),
            "error": None,
        }

        # Add format-specific fields
        if "text" in llm_result:
            response_data["text"] = llm_result["text"]
        if "speakers" in llm_result:
            response_data["speakers"] = llm_result["speakers"]
        if "blocks" in llm_result:
            response_data["blocks"] = llm_result["blocks"]

        # Save transcription to disk
        try:
            # Check if the audio file path is a YouTube URL and extract title for storage
            media_filename = request.audio_file_path
            if audio_downloader._validate_url(request.audio_file_path):
                # Extract video title for YouTube URLs
                video_title = audio_downloader._extract_video_title(request.audio_file_path)
                if video_title.strip():
                    media_filename = video_title
                    logger.info(f"Using video title for storage: {video_title}")
                else:
                    logger.warning("Could not extract video title, using URL")

            saved_path = transcription_service.save_transcription_to_disk(
                media_filename=media_filename,
                transcription_result=result,
                llm_result=llm_result
            )
            if saved_path:
                logger.info(f"Transcription saved to: {saved_path}")
        except Exception as e:
            logger.warning(f"Failed to save transcription to disk: {str(e)}")

        return LLMTranscriptionResponse(**response_data)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/transcribe-youtube", response_model=TranscriptionResponse)
async def transcribe_youtube(request: YouTubeTranscriptionRequest):
    """
    Download audio from YouTube URL and transcribe with optional speaker diarization
    """
    log_action(f"Received YouTube transcription request for URL: {request.youtube_url}")
    logger.info(f"Received YouTube transcription request for: {request.youtube_url}")
    logger.debug(
        f"Request parameters: min_speakers={request.min_speakers}, max_speakers={request.max_speakers}, batch_size={request.batch_size}, verbose={request.verbose}"
    )

    try:
        # Step 1: Download audio from YouTube
        logger.info(f"Downloading audio from YouTube: {request.youtube_url}")
        download_result = audio_downloader.download_audio(
            youtube_url=request.youtube_url, verbose=request.verbose
        )

        audio_path = download_result["audio_path"]
        video_id = download_result["video_id"]
        video_title = download_result.get("title", "")
        file_size = download_result["file_size"]

        logger.info(f"Audio downloaded successfully: {audio_path} ({file_size} bytes)")

        # Step 2: Transcribe the downloaded audio
        logger.info(f"Starting transcription for video: {video_id}")
        result = transcription_service.transcribe_audio(
            audio_path=audio_path,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            batch_size=request.batch_size,
            verbose=request.verbose,
        )

        # Add download metadata to the response
        result_metadata = result.get("metadata", {})
        result_metadata.update(
            {
                "video_id": video_id,
                "download_file_size": file_size,
                "audio_path": audio_path,
            }
        )

        logger.info(f"Transcription completed successfully for video: {video_id}")

        return TranscriptionResponse(
            success=True,
            segments=result.get("segments", []),
            language=result.get("language"),
            metadata=result_metadata,
            error=None,
        )

    except ValueError as e:
        # Invalid URL or similar validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Download or transcription failures
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/transcribe-youtube-llm", response_model=LLMTranscriptionResponse)
async def transcribe_youtube_llm(request: YouTubeLLMTranscriptionRequest):
    """
    Download audio from YouTube URL and transcribe for LLM consumption
    """
    log_action(f"Received YouTube LLM transcription request for URL: {request.youtube_url}")
    logger.info(f"Received YouTube LLM transcription request for: {request.youtube_url}")
    logger.debug(
        f"Request parameters: min_speakers={request.min_speakers}, max_speakers={request.max_speakers}, "
        f"batch_size={request.batch_size}, output_format={request.output_format}, "
        f"remove_filler_words={request.remove_filler_words}, verbose={request.verbose}"
    )

    try:
        # Step 1: Download audio from YouTube
        logger.info(f"Downloading audio from YouTube: {request.youtube_url}")
        download_result = audio_downloader.download_audio(
            youtube_url=request.youtube_url, verbose=request.verbose
        )

        audio_path = download_result["audio_path"]
        video_id = download_result["video_id"]
        video_title = download_result.get("title", "")
        file_size = download_result["file_size"]

        logger.info(f"Audio downloaded successfully: {audio_path} ({file_size} bytes)")

        # Step 2: Transcribe the downloaded audio
        logger.info(f"Starting transcription for video: {video_id}")
        result = transcription_service.transcribe_audio(
            audio_path=audio_path,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            batch_size=request.batch_size,
            verbose=request.verbose,
        )

        # Step 3: Format for LLM consumption
        llm_result = transcription_service.format_for_llm(
            transcription_result=result,
            output_format=request.output_format,
            include_speakers=(request.output_format in ["speaker", "structured", "markdown"]),
            merge_consecutive_speakers=request.merge_consecutive_speakers,
            remove_filler_words=request.remove_filler_words
        )

        # Add download metadata to the response
        llm_metadata = llm_result.get("metadata", {})
        llm_metadata.update(
            {
                "video_id": video_id,
                "download_file_size": file_size,
                "audio_path": audio_path,
            }
        )

        logger.info(f"LLM transcription completed successfully for video: {video_id}")

        # Build response based on format
        response_data = {
            "success": True,
            "language": result.get("language"),
            "metadata": llm_metadata,
            "error": None,
        }

        # Add format-specific fields
        if "text" in llm_result:
            response_data["text"] = llm_result["text"]
        if "speakers" in llm_result:
            response_data["speakers"] = llm_result["speakers"]
        if "blocks" in llm_result:
            response_data["blocks"] = llm_result["blocks"]

        # Save transcription to disk
        try:
            # Use video title for storage instead of URL, fallback to URL if no title
            storage_name = video_title if video_title.strip() else request.youtube_url
            saved_path = transcription_service.save_transcription_to_disk(
                media_filename=storage_name,
                transcription_result=result,
                llm_result=llm_result
            )
            if saved_path:
                logger.info(f"Transcription saved to: {saved_path}")
        except Exception as e:
            logger.warning(f"Failed to save transcription to disk: {str(e)}")

        return LLMTranscriptionResponse(**response_data)

    except ValueError as e:
        # Invalid URL or similar validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Download or transcription failures
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/transcribe_file")
async def transcribe_file(
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    batch_size: Optional[int] = Form(None),
    verbose: bool = Form(True),
):
    """
    Transcribe an uploaded video file

    Args:
        file: Video file to transcribe
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        batch_size: Batch size for processing
        verbose: Enable verbose output

    Returns:
        Transcription results
    """
    logger.info(f"Received file upload request: {file.filename}")

    # Create temporary file to store the upload
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Saved uploaded file to: {temp_file_path}")

        # Extract audio from video file
        audio_path = audio_downloader.extract_audio_from_file(temp_file_path)
        logger.info(f"Extracted audio to: {audio_path}")

        # Perform transcription
        result = transcription_service.transcribe_audio(
            audio_path=audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            batch_size=batch_size,
            verbose=verbose,
        )

        logger.info("Transcription completed successfully")

        # Clean up temporary files
        try:
            os.remove(audio_path)
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

        return {
            "success": True,
            "segments": result["segments"],
            "language": result.get("language"),
            "metadata": {
                "filename": file.filename,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "batch_size": batch_size,
            },
        }

    except Exception as e:
        logger.error(f"Failed to transcribe file: {str(e)}")
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe-file-llm")
async def transcribe_file_llm(
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    batch_size: Optional[int] = Form(None),
    output_format: str = Form("simple"),
    remove_filler_words: bool = Form(False),
    merge_consecutive_speakers: bool = Form(True),
    verbose: bool = Form(True),
):
    """
    Transcribe an uploaded video file and format for LLM consumption

    Args:
        file: Video file to transcribe
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        batch_size: Batch size for processing
        output_format: Format type ("simple", "speaker", "structured", "markdown")
        remove_filler_words: Whether to remove common filler words
        merge_consecutive_speakers: Whether to merge consecutive segments from same speaker
        verbose: Enable verbose output

    Returns:
        LLM-optimized transcription results
    """
    logger.info(f"Received file upload LLM request: {file.filename}")

    # Create temporary file to store the upload
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Saved uploaded file to: {temp_file_path}")

        # Extract audio from video file
        audio_path = audio_downloader.extract_audio_from_file(temp_file_path)
        logger.info(f"Extracted audio to: {audio_path}")

        # Perform transcription
        result = transcription_service.transcribe_audio(
            audio_path=audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Format for LLM consumption
        llm_result = transcription_service.format_for_llm(
            transcription_result=result,
            output_format=output_format,
            include_speakers=(output_format in ["speaker", "structured", "markdown"]),
            merge_consecutive_speakers=merge_consecutive_speakers,
            remove_filler_words=remove_filler_words
        )

        logger.info("LLM transcription completed successfully")

        # Clean up temporary files
        try:
            os.remove(audio_path)
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

        # Add file metadata to the response
        llm_metadata = llm_result.get("metadata", {})
        llm_metadata.update(
            {
                "filename": file.filename,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "batch_size": batch_size,
                "output_format": output_format,
            }
        )

        # Build response based on format
        response_data = {
            "success": True,
            "language": result.get("language"),
            "metadata": llm_metadata,
            "error": None,
        }

        # Add format-specific fields
        if "text" in llm_result:
            response_data["text"] = llm_result["text"]
        if "speakers" in llm_result:
            response_data["speakers"] = llm_result["speakers"]
        if "blocks" in llm_result:
            response_data["blocks"] = llm_result["blocks"]

        # Save transcription to disk
        try:
            saved_path = transcription_service.save_transcription_to_disk(
                media_filename=file.filename,
                transcription_result=result,
                llm_result=llm_result
            )
            if saved_path:
                logger.info(f"Transcription saved to: {saved_path}")
        except Exception as e:
            logger.warning(f"Failed to save transcription to disk: {str(e)}")

        return response_data

    except Exception as e:
        logger.error(f"Failed to transcribe file for LLM: {str(e)}")
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-metadata")
async def extract_metadata(request: MetadataExtractionRequest):
    """
    Extract metadata from YouTube URL without downloading the video
    """
    log_action(f"Received metadata extraction request for URL: {request.youtube_url}")
    logger.info(f"Received metadata extraction request for: {request.youtube_url}")

    try:
        import yt_dlp

        # Validate URL first
        if not audio_downloader._validate_url(request.youtube_url):
            raise ValueError(f"Invalid YouTube URL: {request.youtube_url}")

        video_id = audio_downloader._extract_video_id(request.youtube_url)
        logger.info(f"Extracting metadata for video: {video_id}")

        # Enhanced yt-dlp options for comprehensive metadata extraction
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "skip_download": True,
        }

        # Try cookies file first, fallback to other methods
        try:
            cookie_path = Path("/app/cookies.txt")
            if (
                cookie_path.exists() and cookie_path.stat().st_size > 100
            ):  # Check if cookies file has content
                ydl_opts["cookiefile"] = "/app/cookies.txt"
                logger.info("Using cookies file for authentication")
            else:
                # Fallback to impersonate option
                ydl_opts["impersonate"] = "chrome-131"
                logger.info("Using Chrome impersonation for authentication")
        except Exception as e:
            logger.warning(f"Cookie file check failed, using impersonate: {e}")
            ydl_opts["impersonate"] = "chrome-131"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract full info dict
            info = ydl.extract_info(request.youtube_url, download=False)

            # Extract comprehensive metadata
            metadata = {
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "upload_date": info.get("upload_date"),
                "uploader": info.get("uploader"),
                "uploader_id": info.get("uploader_id"),
                "channel": info.get("channel"),
                "channel_id": info.get("channel_id"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "dislike_count": info.get("dislike_count"),
                "average_rating": info.get("average_rating"),
                "description": info.get("description"),
                "categories": info.get("categories", []),
                "tags": info.get("tags", []),
                "thumbnail": info.get("thumbnail"),
                "thumbnails": info.get("thumbnails", []),
                "webpage_url": info.get("webpage_url"),
                "language": info.get("language"),
                "subtitles": list(info.get("subtitles", {}).keys()),
                "automatic_captions": list(info.get("automatic_captions", {}).keys()),
                "is_live": info.get("is_live", False),
                "was_live": info.get("was_live", False),
                "live_status": info.get("live_status"),
                "release_timestamp": info.get("release_timestamp"),
                "format": info.get("format"),
                "format_id": info.get("format_id"),
                "ext": info.get("ext"),
                "resolution": info.get("resolution"),
                "fps": info.get("fps"),
                "vcodec": info.get("vcodec"),
                "acodec": info.get("acodec"),
                "filesize": info.get("filesize"),
                "age_limit": info.get("age_limit"),
                "chapters": info.get("chapters", []),
                "location": info.get("location"),
                "playlist": info.get("playlist"),
                "playlist_index": info.get("playlist_index"),
                "display_id": info.get("display_id"),
                "fulltitle": info.get("fulltitle"),
                "duration_string": info.get("duration_string"),
                "requested_subtitles": info.get("requested_subtitles"),
                "has_drm": info.get("has_drm"),
                "epoch": info.get("epoch"),
                "n_entries": info.get("n_entries"),
                "playlist_id": info.get("playlist_id"),
                "playlist_title": info.get("playlist_title"),
                "playlist_uploader": info.get("playlist_uploader"),
                "playlist_uploader_id": info.get("playlist_uploader_id"),
            }

            # Remove None values for cleaner response
            metadata = {k: v for k, v in metadata.items() if v is not None}

            logger.info(f"Successfully extracted metadata for video: {video_id}")

            return {"success": True, "metadata": metadata, "error": None}

    except ImportError:
        error_msg = "yt-dlp not available for metadata extraction"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_msg = f"Failed to extract metadata: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "YT-LLM Transcription Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe",
            "transcribe-youtube": "/transcribe-youtube",
            "transcribe_file": "/transcribe_file",
            "transcribe-llm": "/transcribe-llm",
            "transcribe-youtube-llm": "/transcribe-youtube-llm",
            "transcribe-file-llm": "/transcribe-file-llm",
            "extract-metadata": "/extract-metadata",
            "docs": "/docs",
        },
        "llm_endpoints": {
            "description": "Endpoints that format transcription data for LLM consumption",
            "formats": ["simple", "speaker", "structured", "markdown"],
            "features": ["filler word removal", "speaker merging", "clean text output"]
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level=config.LOG_LEVEL.lower())
