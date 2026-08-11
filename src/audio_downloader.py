"""
Audio downloader service for YouTube videos using yt-dlp
"""

import os
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class VideoContext:
    """Metadata for a YouTube video, fetched in one extract_info() call."""
    video_id: str = ""
    title: str = ""
    description: str = ""   # truncated to 500 chars
    channel: str = ""
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


class AudioDownloader:
    """Service for downloading audio from YouTube videos using yt-dlp"""

    def __init__(self, temp_dir: str = "/app/tmp"):
        """
        Initialize audio downloader

        Args:
            temp_dir: Directory for temporary audio files
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Ensure yt-dlp is available
        self._ensure_yt_dlp()

    def _ensure_yt_dlp(self):
        """Ensure yt-dlp is available"""
        if not shutil.which("yt-dlp"):
            raise RuntimeError(
                "yt-dlp not found in container. Please ensure it's installed."
            )
        logger.info("yt-dlp found and ready")

    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)",
            r"youtube\.com/v/([^&\n?#]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        raise ValueError(f"Could not extract video ID from URL: {url}")

    def _validate_url(self, url: str) -> bool:
        """Validate if URL is a valid YouTube URL"""
        youtube_patterns = [
            r"^https?://(www\.)?(youtube\.com|youtu\.be)/",
            r"^https?://m\.youtube\.com/",
        ]

        return any(re.match(pattern, url) for pattern in youtube_patterns)

    def get_video_context(self, youtube_url: str) -> "VideoContext":
        """
        Fetch YouTube metadata in a single extract_info() call.

        Returns a VideoContext dataclass. Never raises — all fields default to
        empty on any error so audio download can proceed regardless.

        Auth: cookies-only (no impersonate — Python API crashes on that option
        in yt-dlp 2026.02.21).
        """
        try:
            import yt_dlp

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
            }
            cookie_path = Path("/app/cookies.txt")
            if cookie_path.exists() and cookie_path.stat().st_size > 100:
                ydl_opts["cookiefile"] = str(cookie_path)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)

            description = (info.get("description") or "")[:500]
            return VideoContext(
                video_id=info.get("id") or self._extract_video_id(youtube_url),
                title=info.get("title") or "",
                description=description,
                channel=info.get("channel") or info.get("uploader") or "",
                tags=info.get("tags") or [],
                categories=info.get("categories") or [],
            )

        except Exception as e:
            logger.warning(f"get_video_context() failed (non-fatal): {e}")
            try:
                video_id = self._extract_video_id(youtube_url)
            except Exception:
                video_id = ""
            return VideoContext(video_id=video_id)

    def download_audio(self, youtube_url: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Download audio from YouTube URL

        Args:
            youtube_url: YouTube video URL
            verbose: Enable verbose logging

        Returns:
            Dict containing audio_path, video_id, title, and metadata

        Raises:
            ValueError: If URL is invalid
            RuntimeError: If download fails
        """
        if not self._validate_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")

        video_id = self._extract_video_id(youtube_url)
        logger.info(f"Downloading audio for video: {video_id}")

        # Fetch all metadata in one call
        ctx = self.get_video_context(youtube_url)
        video_title = ctx.title
        logger.info(f"Video title: {video_title}")

        # Clean up any existing files for this video
        self._cleanup_old_files(video_id)

        # Prepare yt-dlp command
        output_template = str(self.temp_dir / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "0",  # Best quality
            "--output",
            output_template,
            "--no-warnings",
        ]

        # Auth: cookies when available, otherwise nothing.
        #
        # Deliberately no --impersonate: every target needs a working curl_cffi,
        # and when that is missing yt-dlp aborts outright ("Impersonate target
        # ... is not available") rather than falling back — so hardcoding one
        # turns an unauthenticated download that would have worked into a hard
        # failure. get_video_context() went cookies-only for the same reason.
        try:
            cookie_path = Path("/app/cookies.txt")
            if (
                cookie_path.exists() and cookie_path.stat().st_size > 100
            ):  # Check if cookies file has content
                cmd.extend(["--cookies", "/app/cookies.txt"])
                logger.info("Using cookies file for authentication")
            else:
                logger.info("No cookies file — downloading unauthenticated")
        except Exception as e:
            logger.warning(f"Cookie file check failed, downloading unauthenticated: {e}")

        if not verbose:
            cmd.append("--quiet")
        else:
            logger.info(f"Running command: {' '.join(cmd)}")

        cmd.append(youtube_url)

        try:
            # Run yt-dlp command
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=str(self.temp_dir)
            )

            if verbose and result.stdout:
                logger.info(f"yt-dlp output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            error_msg = f"yt-dlp failed: {e.stderr.strip()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Check if audio file was created
        audio_path = self.temp_dir / f"{video_id}.mp3"

        if not audio_path.exists():
            error_msg = f"Expected audio file not found: {audio_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Get file size for validation
        file_size = audio_path.stat().st_size
        logger.info(f"Audio downloaded successfully: {audio_path} ({file_size} bytes)")

        return {
            "audio_path": str(audio_path),
            "video_id": video_id,
            "title": video_title,
            "file_size": file_size,
            "temp_dir": str(self.temp_dir),
            "video_context": ctx,
        }

    def _cleanup_old_files(self, video_id: str):
        """Clean up old audio files for the same video"""
        patterns = [f"{video_id}.*"]

        for pattern in patterns:
            for file_path in self.temp_dir.glob(pattern):
                try:
                    file_path.unlink()
                    logger.debug(f"Cleaned up old file: {file_path}")
                except OSError as e:
                    logger.warning(f"Could not remove {file_path}: {e}")

    def cleanup_all(self):
        """Clean up all temporary files"""
        try:
            for file_path in self.temp_dir.glob("*.mp3"):
                file_path.unlink()
                logger.debug(f"Cleaned up: {file_path}")
        except OSError as e:
            logger.warning(f"Error during cleanup: {e}")

    def get_temp_dir(self) -> str:
        """Get the temporary directory path"""
        return str(self.temp_dir)

    def extract_audio_from_file(self, video_path: str) -> str:
        """
        Extract audio from a video file

        Args:
            video_path: Path to video file

        Returns:
            Path to extracted audio file

        Raises:
            RuntimeError: If extraction fails
        """
        logger.info(f"Extracting audio from file: {video_path}")

        # Generate output path
        video_file = Path(video_path)
        audio_filename = f"{video_file.stem}_audio.mp3"
        audio_path = self.temp_dir / audio_filename

        # Use ffmpeg to extract audio
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "mp3",
            "-ab",
            "192k",  # Audio bitrate
            "-ar",
            "44100",  # Sample rate
            "-y",  # Overwrite output
            str(audio_path),
        ]

        try:
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if not audio_path.exists():
                raise RuntimeError(f"Audio file was not created at {audio_path}")

            logger.info(f"Audio extracted successfully to: {audio_path}")
            return str(audio_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {e}")
            raise RuntimeError(f"Failed to extract audio: {str(e)}")
