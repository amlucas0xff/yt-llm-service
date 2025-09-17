#!/bin/bash
set -e

# Function to start LLM service
start_llm() {
    echo "Starting YT-LLM Transcription Service..."
    
    # Update yt-dlp to latest version
    echo "Updating yt-dlp to latest version..."
    python3 -m pip install --upgrade yt-dlp
    
    exec python3 src/run_llm_api.py
}

# Function to start API server (legacy - for compatibility)
start_api() {
    echo "Starting YT-LLM Transcription Service (legacy API command)..."
    exec python3 src/run_llm_api.py
}

# Function to start MCP server
start_mcp() {
    echo "Starting MCP Server for YouTube Transcription..."
    cd /app
    exec python3 simple-mcp-server/run.py
}

# Function to run CLI (disabled in LLM service)
run_cli() {
    echo "CLI not available in LLM service container"
    exit 1
}

# Function to show help
show_help() {
    cat << HELP
YT-LLM Transcription Service Docker Container

Usage:
  docker run [OPTIONS] yt-llm-service:latest [COMMAND] [ARGS...]

Commands:
  llm                    Start the LLM Transcription Service (default)
  api                    Start the LLM Transcription Service (legacy)
  bash                   Start an interactive bash shell
  help                   Show this help message

Examples:
  # Start LLM service (default)
  docker run -p 8002:8002 --gpus all yt-llm-service:latest

  # Interactive shell
  docker run -it --gpus all yt-llm-service:latest bash

Environment Variables:
  DEVICE                 Device to use (cuda/cpu) - default: cuda
  COMPUTE_TYPE          Computation type (float16/float32/int8) - default: float16
  WHISPER_MODEL         Whisper model size (tiny/base/small/medium/large-v3-turbo) - default: large-v3-turbo
  BATCH_SIZE            Batch size for processing - default: 16
  HF_TOKEN              HuggingFace token for diarization (optional)
  TEMP_DIR              Temporary files directory - default: /app/tmp
  OUTPUT_DIR            Output directory - default: /app/output
  LOG_LEVEL             Logging level - default: INFO

HELP
}

# Main command dispatcher
case "${1:-llm}" in
    llm)
        start_llm
        ;;
    api)
        start_api
        ;;
    mcp)
        start_mcp
        ;;
    cli)
        shift
        run_cli "$@"
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