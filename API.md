# API Documentation

Complete reference for the YT-LLM Transcription Service API endpoints.

## Base URL
```
http://localhost:8002
```

## Authentication
No authentication required for current version. Future versions may include API key authentication.

## Response Format
All endpoints return JSON responses with the following structure:

### Success Response
```json
{
  "success": true,
  "data": "...",  // Endpoint-specific data
  "metadata": {   // Additional information
    "processing_time": 45.2,
    "model_used": "large-v3-turbo"
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error description",
  "detail": "Detailed error message"
}
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check service status and available resources.

**Response:**
```json
{
  "status": "ok",
  "service": "yt-llm-transcription",
  "device_info": {
    "device": "cuda",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_memory": "24564MB",
    "cuda_version": "12.2"
  }
}
```

### 2. YouTube Transcription (Standard)

**POST** `/transcribe-youtube`

Basic YouTube transcription with raw segment output.

**Request Body:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "min_speakers": 1,
  "max_speakers": 5,
  "batch_size": 16,
  "verbose": true
}
```

**Response:**
```json
{
  "success": true,
  "segments": [
    {
      "text": " Never gonna give you up, never gonna let you down",
      "start": 0.5,
      "end": 4.2,
      "speaker": "SPEAKER_00"
    }
  ],
  "language": "en",
  "metadata": {
    "video_id": "dQw4w9WgXcQ",
    "duration": 212.3,
    "download_file_size": 8495616,
    "speakers_detected": 1
  }
}
```

### 3. YouTube Transcription (LLM-Optimized)

**POST** `/transcribe-youtube-llm`

Advanced transcription with LLM-optimized formatting.

**Request Body:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "min_speakers": 1,
  "max_speakers": 3,
  "batch_size": 16,
  "output_format": "markdown",
  "remove_filler_words": true,
  "merge_consecutive_speakers": true,
  "verbose": true
}
```

**Output Formats:**

#### Simple Format
```json
{
  "success": true,
  "text": "Never gonna give you up, never gonna let you down, never gonna run around and desert you.",
  "language": "en",
  "metadata": {
    "word_count": 17,
    "format": "simple"
  }
}
```

#### Speaker Format
```json
{
  "success": true,
  "speakers": {
    "SPEAKER_00": "Never gonna give you up, never gonna let you down.",
    "SPEAKER_01": "That's a great song!"
  },
  "language": "en",
  "metadata": {
    "speaker_count": 2,
    "format": "speaker"
  }
}
```

#### Structured Format
```json
{
  "success": true,
  "blocks": [
    {
      "speaker": "SPEAKER_00",
      "text": "Never gonna give you up",
      "start_time": 0.5,
      "end_time": 2.1,
      "confidence": 0.95
    }
  ],
  "language": "en",
  "metadata": {
    "total_blocks": 25,
    "format": "structured"
  }
}
```

#### Markdown Format
```json
{
  "success": true,
  "text": "# Transcription\n\n## Speaker 1\nNever gonna give you up, never gonna let you down.\n\n## Speaker 2\nThat's a great song!",
  "language": "en",
  "metadata": {
    "format": "markdown"
  }
}
```

### 4. File Upload Transcription

**POST** `/transcribe_file`

Upload and transcribe video/audio files.

**Request:** `multipart/form-data`
- `file`: Video/audio file (mp4, mp3, wav, etc.)
- `min_speakers`: (optional) Minimum speakers
- `max_speakers`: (optional) Maximum speakers
- `batch_size`: (optional) Processing batch size
- `verbose`: (optional) Enable verbose output

**cURL Example:**
```bash
curl -X POST "http://localhost:8002/transcribe_file" \
  -F "file=@/path/to/video.mp4" \
  -F "min_speakers=1" \
  -F "max_speakers=3" \
  -F "verbose=true"
```

### 5. File Upload Transcription (LLM-Optimized)

**POST** `/transcribe-file-llm`

Upload and transcribe with LLM formatting.

**Request:** `multipart/form-data`
- `file`: Video/audio file
- `output_format`: "simple", "speaker", "structured", or "markdown"
- `remove_filler_words`: Boolean
- `merge_consecutive_speakers`: Boolean
- Additional parameters as above

### 6. Audio File Transcription

**POST** `/transcribe`

Direct audio file transcription (file must exist on server).

**Request Body:**
```json
{
  "audio_file_path": "/app/uploads/audio.wav",
  "min_speakers": 1,
  "max_speakers": 3,
  "batch_size": 16,
  "verbose": true
}
```

### 7. Audio File Transcription (LLM-Optimized)

**POST** `/transcribe-llm`

LLM-optimized transcription for existing audio files.

**Request Body:**
```json
{
  "audio_file_path": "/app/uploads/audio.wav",
  "output_format": "speaker",
  "remove_filler_words": true,
  "merge_consecutive_speakers": true,
  "verbose": true
}
```

### 8. YouTube Metadata Extraction

**POST** `/extract-metadata`

Extract video metadata without transcription.

**Request Body:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

**Response:**
```json
{
  "success": true,
  "metadata": {
    "id": "dQw4w9WgXcQ",
    "title": "Rick Astley - Never Gonna Give You Up (Official Video)",
    "duration": 212,
    "uploader": "Rick Astley",
    "view_count": 1234567890,
    "upload_date": "20091025",
    "description": "The official video for...",
    "language": "en",
    "thumbnails": [...],
    "categories": ["Music"],
    "tags": ["rick astley", "never gonna give you up"]
  }
}
```

## Error Handling

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - File or URL not accessible |
| 500 | Internal Server Error - Processing failure |

### Error Examples

#### Invalid YouTube URL
```json
{
  "detail": "Invalid YouTube URL: https://example.com/invalid"
}
```

#### File Not Found
```json
{
  "detail": "Audio file not found: /path/to/missing.wav"
}
```

#### Processing Error
```json
{
  "detail": "Failed to download audio: Video is private"
}
```

## Rate Limiting

Current implementation has no rate limiting. For production use, consider implementing:
- Request rate limiting (e.g., 10 requests/minute)
- Concurrent processing limits
- File size restrictions

## Best Practices

### 1. Optimal Parameters

```json
{
  "batch_size": 16,        // Good balance for most GPUs
  "output_format": "simple", // Fastest processing
  "verbose": false         // Reduces response size
}
```

### 2. Speaker Diarization

```json
{
  "min_speakers": 1,       // Always include
  "max_speakers": 5,       // Reasonable upper bound
  "merge_consecutive_speakers": true  // Cleaner output
}
```

### 3. Large Files

For files > 1 hour:
```json
{
  "batch_size": 8,         // Reduce memory usage
  "output_format": "simple", // Faster processing
  "verbose": false
}
```

## SDKs and Examples

### Python Client Example

```python
import requests
import json

# YouTube transcription
response = requests.post(
    'http://localhost:8002/transcribe-youtube-llm',
    json={
        'youtube_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'output_format': 'markdown',
        'remove_filler_words': True
    }
)

result = response.json()
if result['success']:
    print(result['text'])
else:
    print(f"Error: {result['error']}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

async function transcribeYouTube(url) {
  try {
    const response = await axios.post('http://localhost:8002/transcribe-youtube-llm', {
      youtube_url: url,
      output_format: 'simple',
      remove_filler_words: true
    });

    return response.data;
  } catch (error) {
    console.error('Transcription failed:', error.response.data);
    throw error;
  }
}
```

### cURL Examples

```bash
# Basic YouTube transcription
curl -X POST "http://localhost:8002/transcribe-youtube" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'

# LLM-optimized with speaker separation
curl -X POST "http://localhost:8002/transcribe-youtube-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "output_format": "speaker",
    "remove_filler_words": true,
    "merge_consecutive_speakers": true
  }'

# File upload
curl -X POST "http://localhost:8002/transcribe-file-llm" \
  -F "file=@video.mp4" \
  -F "output_format=markdown" \
  -F "remove_filler_words=true"
```

## Monitoring and Debugging

### Enable Verbose Logging

Set environment variable:
```bash
LOG_LEVEL=DEBUG
```

### Monitor GPU Usage

```bash
# During transcription
nvidia-smi -l 1

# Check memory usage
docker exec yt-llm-service nvidia-smi
```

### Health Check Script

```bash
#!/bin/bash
response=$(curl -s http://localhost:8002/health)
status=$(echo $response | jq -r '.status')

if [ "$status" = "ok" ]; then
    echo "Service is healthy"
    exit 0
else
    echo "Service is unhealthy: $response"
    exit 1
fi
```

---

For more examples and updates, see the [GitHub repository](https://github.com/yourusername/yt-llm-service).