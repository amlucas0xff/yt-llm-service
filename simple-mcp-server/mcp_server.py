"""
FastMCP wrapper for the YouTube transcription service.
Reuses existing services from ../src/ without code duplication.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add both parent directory and src directory to path
parent_dir = os.path.join(os.path.dirname(__file__), '..')
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

from fastmcp import FastMCP
from pydantic import Field

# Import existing services (now that src is in path, use direct imports)
from config import Config
from transcription_service import TranscriptionService
from audio_downloader import AudioDownloader
from simple_logger import log_action

# Initialize configuration and services
config = Config()
transcription_service = TranscriptionService(config)
audio_downloader = AudioDownloader(temp_dir=config.TEMP_DIR)

# Create FastMCP server
mcp = FastMCP("yt-transcription-service")


@mcp.tool()
async def transcribe_youtube(
    youtube_url: str = Field(description="YouTube URL to transcribe"),
    output_format: str = Field(default="simple", description="Output format: simple, speaker, structured, markdown"),
    min_speakers: Optional[int] = Field(default=None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Field(default=None, description="Maximum number of speakers"),
    remove_filler_words: bool = Field(default=False, description="Remove filler words like 'um', 'uh'"),
    merge_consecutive_speakers: bool = Field(default=True, description="Merge consecutive segments from same speaker"),
    verbose: bool = Field(default=True, description="Enable verbose logging")
) -> Dict[str, Any]:
    """
    Download audio from YouTube URL and transcribe for LLM consumption.

    Returns a dictionary with transcription results including text, speakers (if applicable),
    metadata, and language detection.
    """
    log_action(f"MCP: Received YouTube transcription request for URL: {youtube_url}")

    try:
        # Step 1: Download audio from YouTube
        download_result = audio_downloader.download_audio(
            youtube_url=youtube_url,
            verbose=verbose
        )

        audio_path = download_result["audio_path"]
        video_id = download_result["video_id"]
        video_title = download_result.get("title", "")
        file_size = download_result["file_size"]

        # Step 2: Transcribe the downloaded audio
        transcription_result = transcription_service.transcribe_audio(
            audio_path=audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            batch_size=config.BATCH_SIZE,
            verbose=verbose,
        )

        # Step 3: Format for LLM consumption
        llm_result = transcription_service.format_for_llm(
            transcription_result=transcription_result,
            output_format=output_format,
            include_speakers=(output_format in ["speaker", "structured", "markdown"]),
            merge_consecutive_speakers=merge_consecutive_speakers,
            remove_filler_words=remove_filler_words
        )

        # Add download metadata
        llm_metadata = llm_result.get("metadata", {})
        llm_metadata.update({
            "video_id": video_id,
            "video_title": video_title,
            "download_file_size": file_size,
            "audio_path": audio_path,
            "youtube_url": youtube_url
        })

        # Build response
        response_data = {
            "success": True,
            "language": transcription_result.get("language"),
            "metadata": llm_metadata,
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
            storage_name = video_title if video_title.strip() else youtube_url
            saved_path = transcription_service.save_transcription_to_disk(
                media_filename=storage_name,
                transcription_result=transcription_result,
                llm_result=llm_result
            )
            if saved_path:
                response_data["metadata"]["saved_path"] = str(saved_path)
        except Exception as e:
            response_data["metadata"]["save_warning"] = f"Failed to save: {str(e)}"

        return response_data

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metadata": {"youtube_url": youtube_url}
        }


@mcp.tool()
async def transcribe_file(
    audio_file_path: str = Field(description="Path to audio/video file to transcribe"),
    output_format: str = Field(default="simple", description="Output format: simple, speaker, structured, markdown"),
    min_speakers: Optional[int] = Field(default=None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Field(default=None, description="Maximum number of speakers"),
    remove_filler_words: bool = Field(default=False, description="Remove filler words like 'um', 'uh'"),
    merge_consecutive_speakers: bool = Field(default=True, description="Merge consecutive segments from same speaker"),
    verbose: bool = Field(default=True, description="Enable verbose logging")
) -> Dict[str, Any]:
    """
    Transcribe a local audio/video file for LLM consumption.

    Returns a dictionary with transcription results including text, speakers (if applicable),
    metadata, and language detection.
    """
    log_action(f"MCP: Received file transcription request for: {audio_file_path}")

    try:
        # Validate file exists
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        file_size = audio_path.stat().st_size

        # Transcribe the audio
        transcription_result = transcription_service.transcribe_audio(
            audio_path=str(audio_file_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            batch_size=config.BATCH_SIZE,
            verbose=verbose,
        )

        # Format for LLM consumption
        llm_result = transcription_service.format_for_llm(
            transcription_result=transcription_result,
            output_format=output_format,
            include_speakers=(output_format in ["speaker", "structured", "markdown"]),
            merge_consecutive_speakers=merge_consecutive_speakers,
            remove_filler_words=remove_filler_words
        )

        # Build response
        response_data = {
            "success": True,
            "language": transcription_result.get("language"),
            "metadata": {
                "file_path": audio_file_path,
                "file_size": file_size,
                **llm_result.get("metadata", {})
            },
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
                media_filename=audio_path.stem,
                transcription_result=transcription_result,
                llm_result=llm_result
            )
            if saved_path:
                response_data["metadata"]["saved_path"] = str(saved_path)
        except Exception as e:
            response_data["metadata"]["save_warning"] = f"Failed to save: {str(e)}"

        return response_data

    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "metadata": {"file_path": audio_file_path}
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metadata": {"file_path": audio_file_path}
        }


@mcp.tool()
async def get_health() -> Dict[str, Any]:
    """
    Get the health status and configuration of the transcription service.

    Returns system information, device capabilities, and service configuration.
    """
    try:
        import torch

        device_info = {
            "device": config.DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            device_info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory

        return {
            "status": "healthy",
            "service": "YT-LLM Transcription Service (MCP)",
            "device_info": device_info,
            "model_config": {
                "whisper_model": config.WHISPER_MODEL,
                "compute_type": config.COMPUTE_TYPE,
                "batch_size": config.BATCH_SIZE,
                "default_output_format": config.LLM_OUTPUT_FORMAT
            },
            "directories": {
                "temp_dir": str(config.TEMP_DIR),
                "output_dir": str(config.OUTPUT_DIR)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "service": "YT-LLM Transcription Service (MCP)"
        }


@mcp.resource("transcriptions://list")
async def list_transcriptions() -> str:
    """
    List all saved transcriptions from the output directory.

    Returns a formatted list of available transcription files.
    """
    try:
        transcription_files = []
        output_dir = Path(config.OUTPUT_DIR)

        if output_dir.exists():
            for file_path in output_dir.rglob("*.json"):
                try:
                    stat = file_path.stat()
                    transcription_files.append({
                        "filename": file_path.name,
                        "path": str(file_path.relative_to(output_dir)),
                        "size": stat.st_size,
                        "modified": stat.st_mtime
                    })
                except Exception:
                    continue

        if not transcription_files:
            return "No transcription files found in the output directory."

        # Sort by modification time (newest first)
        transcription_files.sort(key=lambda x: x["modified"], reverse=True)

        result = "Available Transcriptions:\n\n"
        for file_info in transcription_files:
            size_mb = file_info["size"] / (1024 * 1024)
            result += f"• {file_info['filename']} ({size_mb:.2f} MB)\n"
            result += f"  Path: {file_info['path']}\n\n"

        return result

    except Exception as e:
        return f"Error listing transcriptions: {str(e)}"


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run())