"""
Storage service for persistent transcription storage
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StorageService:
    """Service for managing persistent storage of transcriptions"""

    def __init__(self, output_dir: str | Path):
        """
        Initialize storage service

        Args:
            output_dir: Base directory for storing transcriptions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"StorageService initialized with output_dir: {self.output_dir}")

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe filesystem use

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for filesystem
        """
        # For video titles, we don't need to remove extensions like Path.stem does
        # Path.stem treats "/" as path separator and would break video titles
        name = filename

        # Replace spaces with underscores
        sanitized = name.replace(' ', '_')

        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', sanitized)

        # Remove special characters except underscores, hyphens, and periods
        sanitized = re.sub(r'[^\w\-_.]', '', sanitized)

        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')

        # Ensure not empty and not too long
        if not sanitized:
            sanitized = "untitled"
        elif len(sanitized) > 100:
            sanitized = sanitized[:100]

        return sanitized

    def get_directory_path(self, media_filename: str) -> Path:
        """
        Get the directory path for a media file

        Args:
            media_filename: Original media filename

        Returns:
            Path to the directory for this media file
        """
        sanitized_name = self.sanitize_filename(media_filename)
        return self.output_dir / sanitized_name

    def get_next_transcription_filename(self, directory: Path) -> str:
        """
        Get the next available transcription filename in a directory

        Args:
            directory: Directory to check for existing files

        Returns:
            Next available filename (transcription.md, transcription_1.md, etc.)
        """
        base_filename = "transcription.md"
        file_path = directory / base_filename

        if not file_path.exists():
            return base_filename

        # Find next available numbered filename
        counter = 1
        while True:
            filename = f"transcription_{counter}.md"
            file_path = directory / filename
            if not file_path.exists():
                return filename
            counter += 1

    def save_transcription(
        self,
        media_filename: str,
        transcription_text: str,
        metadata: Dict[str, Any],
        language: Optional[str] = None
    ) -> str:
        """
        Save transcription to disk with proper directory structure

        Args:
            media_filename: Original media filename
            transcription_text: Formatted transcription text
            metadata: Metadata from transcription service
            language: Detected language

        Returns:
            Path to saved file

        Raises:
            OSError: If file operations fail
        """
        try:
            # Create directory
            directory = self.get_directory_path(media_filename)
            directory.mkdir(parents=True, exist_ok=True)

            # Get next available filename
            filename = self.get_next_transcription_filename(directory)
            file_path = directory / filename

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create markdown content
            content = self._create_markdown_content(
                media_filename,
                timestamp,
                transcription_text,
                metadata,
                language
            )

            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Transcription saved successfully to: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save transcription for {media_filename}: {str(e)}")
            raise OSError(f"Failed to save transcription: {str(e)}")

    def _create_markdown_content(
        self,
        media_filename: str,
        timestamp: str,
        transcription_text: str,
        metadata: Dict[str, Any],
        language: Optional[str] = None
    ) -> str:
        """
        Create markdown content for transcription file

        Args:
            media_filename: Original media filename
            timestamp: Creation timestamp
            transcription_text: Formatted transcription
            metadata: Metadata object
            language: Detected language

        Returns:
            Formatted markdown content
        """
        # Clean metadata for JSON serialization
        clean_metadata = self._clean_metadata_for_json(metadata)

        # Add language to metadata if available
        if language:
            clean_metadata['detected_language'] = language

        metadata_json = json.dumps(clean_metadata, indent=2, ensure_ascii=False)

        content = f"""# Transcription: {media_filename}

**Date:** {timestamp}

---

## Metadata
```json
{metadata_json}
```

## Transcription
```
{transcription_text}
```
"""
        return content

    def _clean_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata for JSON serialization by removing non-serializable objects

        Args:
            metadata: Original metadata dict

        Returns:
            Cleaned metadata dict safe for JSON serialization
        """
        def clean_value(value):
            if isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(item) for item in value]
            elif isinstance(value, (str, int, float, bool, type(None))):
                return value
            else:
                # Convert non-serializable objects to string
                return str(value)

        return clean_value(metadata)

    def directory_exists(self, media_filename: str) -> bool:
        """
        Check if directory exists for a media file

        Args:
            media_filename: Original media filename

        Returns:
            True if directory exists, False otherwise
        """
        directory = self.get_directory_path(media_filename)
        return directory.exists()

    def list_transcriptions(self, media_filename: str) -> list[str]:
        """
        List all transcription files for a media file

        Args:
            media_filename: Original media filename

        Returns:
            List of transcription file paths
        """
        directory = self.get_directory_path(media_filename)

        if not directory.exists():
            return []

        transcription_files = []
        for file_path in directory.glob("transcription*.md"):
            transcription_files.append(str(file_path))

        return sorted(transcription_files)