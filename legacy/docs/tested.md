# YouTube LLM Transcription Service - Testing Documentation

## Available API Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Purpose**: Check service status and GPU availability
- **Response**: JSON with service status, GPU info, and CUDA availability

### 2. YouTube Transcription
- **Endpoint**: `POST /transcribe-youtube`
- **Purpose**: Download and transcribe YouTube videos with speaker diarization
- **Parameters**:
  - `youtube_url` (required): YouTube video URL
  - `min_speakers` (optional): Minimum expected speakers
  - `max_speakers` (optional): Maximum expected speakers
  - `batch_size` (optional): GPU processing batch size
  - `verbose` (optional): Enable detailed logging

### 3. Other Endpoints
- `GET /` - Service info and available endpoints
- `POST /extract-metadata` - Extract YouTube video metadata
- `POST /transcribe` - Transcribe local audio file
- `POST /transcribe_file` - Upload and transcribe audio file

## Testing Commands Used

### Service Health Check
```bash
curl http://localhost:8002/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "yt-llm-transcription",
  "device_info": {
    "device": "cuda",
    "compute_type": "float16",
    "torch_version": "2.7.1+cu126",
    "cuda_available": true,
    "cuda_version": "12.6",
    "gpu_count": 1,
    "current_gpu": 0,
    "gpu_name": "NVIDIA GeForce RTX 3090 Ti",
    "gpu_memory": {"allocated": 0, "cached": 0}
  }
}
```

### YouTube Transcription Test
```bash
curl -X POST http://localhost:8002/transcribe-youtube \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=NQbbAoP_zEg",
    "min_speakers": 2,
    "max_speakers": 4,
    "verbose": true
  }'
```

### Save Results to File
```bash
curl -X POST http://localhost:8002/transcribe-youtube \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=NQbbAoP_zEg",
    "min_speakers": 2,
    "max_speakers": 4,
    "verbose": true
  }' | jq > transcription_result.json
```

## Troubleshooting Commands

### Docker Service Management
```bash
# Check if service is running
docker ps | grep yt-llm

# View service logs
docker logs yt-llm-service --tail 50

# View logs with specific filters
docker logs yt-llm-service --tail 30 | grep -E "(WhisperX|Loading|Processing|Transcribing|Speaker|segments|Complete|success)"

# Restart service
docker restart yt-llm-service

# Check Python process in container
docker exec yt-llm-service ps aux | grep python
```

### Permission Fixes Required
```bash
# Fix log file permissions (if permission errors occur)
sudo touch ../yt-backend/logs/log.txt
sudo chmod 666 ../yt-backend/logs/log.txt

# Fix temporary directory permissions
sudo chmod 777 ../yt-backend/tmp

# Check directory permissions
ls -la ../yt-backend/logs/
ls -la ../yt-backend/tmp/
```

## Test Results

### Test Video Details
- **URL**: https://www.youtube.com/watch?v=NQbbAoP_zEg
- **Title**: MCP Server Security Tutorial
- **Duration**: 463 seconds (7 min 43 sec)
- **File Size**: 18.5 MB
- **Language**: English (99.9% confidence)

### Performance Metrics
- **Total Processing Time**: ~208.5 seconds
- **Download Time**: ~10 seconds
- **Transcription Time**: ~198 seconds
- **GPU**: NVIDIA GeForce RTX 3090 Ti
- **Model**: WhisperX large-v3-turbo
- **Compute Type**: float16
- **Batch Size**: 16 (default)

### Speaker Diarization Results
- **Requested**: 2-4 speakers
- **Detected**: 1 speaker (SPEAKER_00)
- **Accuracy**: High confidence speaker segmentation

## Prerequisites Discovered

### Required Directory Structure
```
../yt-backend/
├── logs/       # Must be writable (chmod 666)
├── tmp/        # Must be writable (chmod 777)
└── output/     # Must exist
```

### System Requirements
- NVIDIA GPU with CUDA support
- Docker with nvidia runtime
- Sufficient disk space for model downloads (~3GB)
- Network access for YouTube downloads

### First Run Considerations
- WhisperX model download on first use (~3-5 minutes)
- pyannote.audio models for speaker diarization
- May see CUDA/cuDNN warnings (can be ignored)

## Sample Response Structure

### Successful Transcription Response
```json
{
  "success": true,
  "segments": [
    {
      "start": 0.487,
      "end": 1.268,
      "text": " Good morning, everyone.",
      "speaker": "SPEAKER_00",
      "words": [
        {
          "word": "Good",
          "start": 0.487,
          "end": 0.727,
          "score": 0.344
        }
      ]
    }
  ],
  "language": "en",
  "metadata": {
    "video_id": "NQbbAoP_zEg",
    "download_file_size": 18512517,
    "audio_path": "/app/tmp/NQbbAoP_zEg.mp3",
    "audio_duration": 463.0,
    "original_language": "en",
    "language_probability": 0.9990234375,
    "total_speakers_detected": 1,
    "transcription_time": 208.5
  },
  "error": null
}
```

### Error Response Examples
```json
{
  "detail": "yt-dlp failed: ERROR: unable to open for writing: [Errno 13] Permission denied: '/app/tmp/NQbbAoP_zEg.webm.part'"
}
```

## Common Issues and Solutions

1. **Permission Denied Errors**
   - Run the permission fix commands above
   - Ensure Docker volumes are properly mapped

2. **Service Not Responding**
   - Check Docker container status
   - Review logs for startup errors
   - Verify GPU availability

3. **Slow First Request**
   - Normal behavior - models downloading
   - Subsequent requests will be faster

4. **Memory Errors**
   - Reduce batch_size parameter
   - Ensure sufficient GPU memory

## Notes
- Service runs on port 8002
- Uses yt-dlp for YouTube downloads with cookie support
- Includes word-level timestamps with confidence scores
- Supports batch processing with configurable batch_size
- Audio files are temporarily stored in /app/tmp/
- Automatic cleanup after transcription completion