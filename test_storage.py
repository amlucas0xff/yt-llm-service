#!/usr/bin/env python3
"""
Simple test script to verify storage functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from storage_service import StorageService
from transcription_service import TranscriptionService
from config import Config
import tempfile
import shutil
from pathlib import Path


def test_storage_service():
    """Test basic storage service functionality"""
    print("Testing StorageService...")

    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = StorageService(temp_dir)

        # Test filename sanitization
        test_cases = [
            ("test video.mp4", "test_video"),
            ("My<Video>File.mov", "MyVideoFile"),
            ("strange:file|name?.avi", "strangefilename"),
            ("", "untitled"),
            ("a" * 150, "a" * 100),  # Test length limit
        ]

        for original, expected in test_cases:
            result = storage.sanitize_filename(original)
            print(f"  Sanitize '{original}' -> '{result}'")
            if not result.startswith(expected[:20]):  # Approximate check
                print(f"    Warning: Expected to start with '{expected[:20]}'")

        # Test directory creation and file saving
        test_filename = "test_video.mp4"
        test_transcription = "This is a test transcription.\n\nSpeaker 1: Hello world!"
        test_metadata = {
            "duration": 120,
            "language": "en",
            "model": "large-v3-turbo"
        }

        saved_path = storage.save_transcription(
            media_filename=test_filename,
            transcription_text=test_transcription,
            metadata=test_metadata,
            language="en"
        )

        print(f"  Saved test transcription to: {saved_path}")

        # Verify file exists and has content
        if Path(saved_path).exists():
            with open(saved_path, 'r') as f:
                content = f.read()
                if "test transcription" in content and "Speaker 1" in content:
                    print("  ✓ File content looks correct")
                else:
                    print("  ✗ File content doesn't match expected")
        else:
            print("  ✗ Saved file doesn't exist")

        # Test incremental naming
        saved_path2 = storage.save_transcription(
            media_filename=test_filename,
            transcription_text="Second transcription",
            metadata=test_metadata,
            language="en"
        )

        if "transcription_1.md" in saved_path2:
            print("  ✓ Incremental naming works")
        else:
            print(f"  ✗ Expected incremental name, got: {saved_path2}")

        # Test directory listing
        transcriptions = storage.list_transcriptions(test_filename)
        if len(transcriptions) == 2:
            print("  ✓ Directory listing works")
        else:
            print(f"  ✗ Expected 2 transcriptions, found: {len(transcriptions)}")


def test_transcription_service_integration():
    """Test TranscriptionService storage integration"""
    print("\nTesting TranscriptionService integration...")

    # Create config with temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock config without calling the full Config constructor
        class MockConfig:
            def __init__(self):
                self.OUTPUT_DIR = Path(temp_dir)
                self.DEVICE = "cpu"
                self.COMPUTE_TYPE = "int8"
                self.WHISPER_MODEL = "large-v3-turbo"
                self.BATCH_SIZE = 16
                self.HF_TOKEN = None

        config = MockConfig()

        # Create service (without loading actual models)
        service = TranscriptionService(config, use_gpu=False)

        # Mock transcription results
        mock_transcription_result = {
            "segments": [
                {"text": "Hello world", "start": 0.0, "end": 2.0},
                {"text": "This is a test", "start": 2.5, "end": 5.0}
            ],
            "language": "en",
            "metadata": {
                "device_used": "cpu",
                "model": "large-v3-turbo",
                "speakers_detected": 1
            }
        }

        mock_llm_result = {
            "success": True,
            "text": "Hello world. This is a test.",
            "language": "en",
            "metadata": {
                "word_count": 6,
                "format": "simple"
            }
        }

        # Test save_transcription_to_disk
        saved_path = service.save_transcription_to_disk(
            media_filename="test_audio.mp3",
            transcription_result=mock_transcription_result,
            llm_result=mock_llm_result
        )

        if saved_path and Path(saved_path).exists():
            print("  ✓ TranscriptionService storage integration works")

            # Check file content
            with open(saved_path, 'r') as f:
                content = f.read()
                if "Hello world" in content and "test_audio.mp3" in content:
                    print("  ✓ File content includes transcription and filename")
                else:
                    print("  ✗ File content missing expected elements")
        else:
            print("  ✗ TranscriptionService storage failed")

        # Test with speaker format
        mock_llm_result_speakers = {
            "success": True,
            "speakers": {
                "SPEAKER_00": "Hello world",
                "SPEAKER_01": "This is a test"
            },
            "language": "en",
            "metadata": {
                "speaker_count": 2,
                "format": "speaker"
            }
        }

        saved_path2 = service.save_transcription_to_disk(
            media_filename="test_speaker_audio.mp3",
            transcription_result=mock_transcription_result,
            llm_result=mock_llm_result_speakers
        )

        if saved_path2 and Path(saved_path2).exists():
            with open(saved_path2, 'r') as f:
                content = f.read()
                if "Speaker 1:" in content and "Speaker 2:" in content:
                    print("  ✓ Speaker format conversion works")
                else:
                    print("  ✗ Speaker format not correctly converted")


if __name__ == "__main__":
    print("Running storage functionality tests...\n")

    try:
        test_storage_service()
        test_transcription_service_integration()
        print("\n✓ All tests completed!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)