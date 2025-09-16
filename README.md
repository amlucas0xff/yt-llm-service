# YT-LLM Transcription Service

A high-performance FastAPI service for YouTube audio transcription with advanced speaker diarization and LLM-optimized output formatting. Built with WhisperX for state-of-the-art accuracy and CUDA support for scalable processing.

## ✨ Features

### Core Capabilities
- **YouTube Audio Processing**: Direct download and transcription from YouTube URLs
- **File Upload Support**: Transcribe uploaded video/audio files
- **Speaker Diarization**: Advanced speaker separation using pyannote.audio
- **Multiple Output Formats**: Simple text, speaker-separated, structured JSON, and Markdown
- **LLM-Optimized Outputs**: Clean, formatted transcriptions ready for AI processing
- **GPU Acceleration**: CUDA support for high-speed processing
- **Persistent Storage**: Automatic saving and organization of transcriptions

### Advanced Features
- **Filler Word Removal**: Intelligent removal of "um", "uh", and other speech disfluencies
- **Speaker Merging**: Automatic merging of consecutive segments from the same speaker
- **Batch Processing**: Configurable batch sizes for optimal performance
- **RESTful API**: Complete FastAPI implementation with automatic OpenAPI documentation
- **Docker Support**: Containerized deployment with GPU passthrough
- **Health Monitoring**: Built-in health checks and device information endpoints

## 🛠 Tech Stack

- **Framework**: FastAPI with async/await support
- **ML Models**: WhisperX (large-v3-turbo), pyannote.audio for diarization
- **Audio Processing**: yt-dlp for YouTube downloads, ffmpeg for format conversion
- **GPU Support**: CUDA/cuDNN with PyTorch backend
- **Containerization**: Docker with NVIDIA runtime support
- **Language**: Python 3.11+ with type hints throughout

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- NVIDIA GPU with CUDA support (recommended)
- Docker and docker-compose (optional)
- HuggingFace account for model access

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/yt-llm-service.git
cd yt-llm-service
```

2. **Set up environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Set HF_TOKEN from https://huggingface.co/settings/tokens
```

3. **Install dependencies**
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

4. **Run the service**
```bash
# Development
uvicorn src.run_llm_api:app --host 0.0.0.0 --port 8002 --reload

# Production
python src/run_llm_api.py
```

### Docker Deployment

1. **Prepare configuration**
```bash
cp docker-compose.example.yml docker-compose.yml
cp .env.example .env
# Edit both files as needed
```

2. **Build and run**
```bash
docker-compose up --build
```

The service will be available at `http://localhost:8002`

## 📋 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

### Core Endpoints

#### 1. YouTube Transcription (LLM-Optimized)
```bash
POST /transcribe-youtube-llm
```

**Example Request:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "output_format": "markdown",
  "remove_filler_words": true,
  "merge_consecutive_speakers": true,
  "min_speakers": 1,
  "max_speakers": 3
}
```

**Example Response:**
```json
{
  "success": true,
  "text": "# Transcription\n\n**Speaker 1:** Never gonna give you up, never gonna let you down...",
  "language": "en",
  "metadata": {
    "video_id": "dQw4w9WgXcQ",
    "duration": 212,
    "speakers_detected": 1,
    "word_count": 156
  }
}
```

#### 2. File Upload Transcription
```bash
POST /transcribe-file-llm
```

Upload video/audio files directly for transcription.

#### 3. Health Check
```bash
GET /health
```

Returns service status and GPU information.

### Output Formats

- **simple**: Clean text without speaker labels
- **speaker**: Text with speaker identification
- **structured**: JSON with detailed segment information
- **markdown**: Formatted Markdown with speaker headers

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Processing device (cuda/cpu) |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model size |
| `BATCH_SIZE` | `16` | Processing batch size |
| `HF_TOKEN` | - | HuggingFace API token |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

See `.env.example` for complete configuration options.

### Performance Tuning

#### GPU Memory Optimization
```bash
# For 8GB GPU
BATCH_SIZE=8
COMPUTE_TYPE=float16

# For 24GB GPU
BATCH_SIZE=32
COMPUTE_TYPE=float16

# CPU-only (slower)
DEVICE=cpu
COMPUTE_TYPE=float32
BATCH_SIZE=4
```

## 📁 Project Structure

```
yt-llm-service/
├── src/
│   ├── run_llm_api.py      # FastAPI application
│   ├── transcription_service.py  # Core transcription logic
│   ├── audio_downloader.py       # YouTube/file processing
│   ├── storage_service.py        # Persistent storage
│   └── config.py                 # Configuration management
├── data/
│   ├── output/            # Transcription results
│   ├── tmp/              # Temporary processing files
│   └── logs/             # Application logs
├── tests/
│   └── test_storage.py   # Storage functionality tests
├── docker-compose.yml    # Docker configuration
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🧪 Testing

### Run Tests
```bash
# Storage functionality
python test_storage.py

# API endpoints
python test_llm_endpoints.py

# Full test suite
pytest tests/
```

### Example Test Usage
```bash
# Test YouTube transcription
curl -X POST "http://localhost:8002/transcribe-youtube-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "output_format": "simple"
  }'
```

## 🐳 Docker Configuration

### Requirements
- Docker Engine 20.10+
- docker-compose v2.0+
- NVIDIA Container Toolkit (for GPU support)

### GPU Setup
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 📊 Performance Benchmarks

| Model | GPU | Batch Size | Processing Speed | Accuracy |
|-------|-----|------------|------------------|----------|
| large-v3-turbo | RTX 4090 | 32 | ~10x realtime | Excellent |
| large-v3-turbo | RTX 3080 | 16 | ~6x realtime | Excellent |
| large-v3 | RTX 4090 | 16 | ~4x realtime | Superior |
| CPU-only | Intel i9 | 4 | ~0.5x realtime | Good |

*Benchmarks based on typical YouTube content (10-minute videos)*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for advanced transcription capabilities
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube processing
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📞 Support

For issues and questions:
- Open an [issue](https://github.com/yourusername/yt-llm-service/issues)
- Check existing [documentation](http://localhost:8002/docs)
- Review [configuration examples](.env.example)

---

**Built with ❤️ for the AI/ML community**