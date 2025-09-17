#!/bin/bash

# Initialize data directories for the YT-LLM service
# This script ensures all required directories exist with proper structure

echo "Initializing data directories for YT-LLM service..."

# Create main data directories
directories=(
    "data/uploads"
    "data/tmp"
    "data/output"
    "data/logs"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    else
        echo "Directory already exists: $dir"
    fi
done

# Create .gitkeep files to preserve directory structure in git
for dir in "${directories[@]}"; do
    gitkeep="$dir/.gitkeep"
    if [ ! -f "$gitkeep" ]; then
        touch "$gitkeep"
        echo "Created .gitkeep in: $dir"
    fi
done

echo "Data directory initialization complete!"
echo ""
echo "Directory structure:"
echo "  data/"
echo "  ├── uploads/  # Place files here for transcription"
echo "  ├── tmp/      # Temporary processing files"
echo "  ├── output/   # Transcription results"
echo "  └── logs/     # Application logs"
echo ""
echo "To upload a file for transcription:"
echo "  1. Place your MP4/audio file in: data/uploads/"
echo "  2. Use curl with the file path:"
echo "     curl -X POST 'http://localhost:8002/transcribe_file' \\"
echo "          -F 'file=@data/uploads/your-video.mp4'"