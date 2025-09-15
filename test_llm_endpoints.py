#!/usr/bin/env python3
"""
Test script for LLM endpoints functionality
"""

import json
import sys
import os

# Add the src directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from transcription_service import TranscriptionService


def test_llm_formatting():
    """Test the LLM formatting functionality using existing transcription result"""

    print("🧪 Testing LLM formatting functionality...")

    # Load the existing transcription result
    with open('transcription_result_contextenginering.json', 'r') as f:
        transcription_data = json.load(f)

    # Set environment variables to use local directories for testing
    os.environ['TEMP_DIR'] = './tmp'
    os.environ['OUTPUT_DIR'] = './output'

    # Initialize services
    config = Config()
    transcription_service = TranscriptionService(config, use_gpu=False)

    # Test different formats
    formats_to_test = ["simple", "speaker", "structured", "markdown"]

    for format_type in formats_to_test:
        print(f"\n📋 Testing {format_type} format...")

        result = transcription_service.format_for_llm(
            transcription_result=transcription_data,
            output_format=format_type,
            remove_filler_words=True,
            merge_consecutive_speakers=True
        )

        print(f"✅ Success: {result['success']}")
        print(f"📊 Metadata: {result['metadata']}")

        if 'text' in result:
            print(f"📝 Text length: {len(result['text'])} characters")
            print(f"📄 Preview: {result['text'][:100]}...")

        if 'speakers' in result:
            print(f"🗣️ Speakers found: {list(result['speakers'].keys())}")

        if 'blocks' in result:
            print(f"📦 Blocks count: {len(result['blocks'])}")

        print("=" * 60)

    print("\n🎉 All LLM formatting tests completed successfully!")


def test_filler_word_removal():
    """Test filler word removal functionality"""

    print("\n🧪 Testing filler word removal...")

    # Set environment variables to use local directories for testing
    os.environ['TEMP_DIR'] = './tmp'
    os.environ['OUTPUT_DIR'] = './output'

    config = Config()
    transcription_service = TranscriptionService(config, use_gpu=False)

    # Create test data with filler words
    test_segments = [
        {"text": "Um, this is a test, uh, with filler words, you know?", "speaker": "SPEAKER_00"},
        {"text": "Hmm, let me think, er, about this problem.", "speaker": "SPEAKER_00"},
        {"text": "Actually, mm, this is working great!", "speaker": "SPEAKER_01"}
    ]

    test_data = {"segments": test_segments, "language": "en"}

    # Test with filler word removal
    result_with_removal = transcription_service.format_for_llm(
        transcription_result=test_data,
        output_format="simple",
        remove_filler_words=True
    )

    # Test without filler word removal
    result_without_removal = transcription_service.format_for_llm(
        transcription_result=test_data,
        output_format="simple",
        remove_filler_words=False
    )

    print(f"📝 Original text: {result_without_removal['text']}")
    print(f"✨ Cleaned text: {result_with_removal['text']}")

    print("\n🎉 Filler word removal test completed!")


if __name__ == "__main__":
    try:
        test_llm_formatting()
        test_filler_word_removal()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)