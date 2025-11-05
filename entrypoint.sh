#!/bin/bash
set -e

# Function to start service
start_service() {
    echo "Starting YT-LLM Transcription Service..."

    # Update yt-dlp to latest version
    echo "Updating yt-dlp to latest version..."
    python3 -m pip install --upgrade yt-dlp

    exec python3 src/run_llm_api.py
}

# Function to show help
show_help() {
    cat << HELP
YT-LLM Transcription Service Docker Container

Usage:
  docker-compose up

  Or manually:
  docker run -p 8002:8002 --gpus all yt-llm-service:latest

Commands:
  bash                   Start an interactive bash shell
  help                   Show this help message

Environment Variables:
  DEVICE                 Device to use (cuda/cpu) - default: cuda
  COMPUTE_TYPE          Computation type (float16/float32/int8) - default: float16
  WHISPER_MODEL         Whisper model size - default: large-v3-turbo
  BATCH_SIZE            Batch size for processing - default: 16
  HF_TOKEN              HuggingFace token (required)
  LOG_LEVEL             Logging level - default: INFO

HELP
}

# Main command dispatcher
case "${1:-llm}" in
    llm|api|"")
        start_service
        ;;
    bash)
        exec /bin/bash
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run 'help' for usage information"
        exit 1
        ;;
esac 