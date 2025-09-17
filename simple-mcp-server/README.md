# FastMCP YouTube Transcription Server

A minimal Model Context Protocol (MCP) wrapper around the existing YouTube transcription FastAPI service. This server exposes the transcription functionality directly to LLMs like Claude through the MCP protocol.

## 🎯 Key Features

- **Zero Code Duplication**: Directly imports and reuses services from `../src/`
- **MCP Tools**: YouTube transcription, file transcription, health checks
- **MCP Resources**: List and access saved transcriptions
- **Instant Integration**: Works with Claude Desktop and other MCP clients
- **Minimal Setup**: Only 4 files, inherits all functionality from parent service

## 🚀 Quick Start

### Prerequisites

1. **Parent Service Setup**: The main YouTube transcription service must be configured first
   ```bash
   cd ..  # Go to parent directory
   uv pip install -r requirements.txt
   # Set up .env file with HF_TOKEN and other config
   ```

2. **MCP Dependencies**: Install minimal dependencies for this wrapper
   ```bash
   cd simple-mcp-server
   pip install -r requirements.txt
   ```

### Running the MCP Server

```bash
# From the simple-mcp-server directory
python run.py
```

The server will start and communicate via stdio (standard input/output), which is how MCP servers work.

## 🔧 Claude Desktop Configuration

Add this configuration to your Claude Desktop settings to use the MCP server:

### macOS
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "yt-transcription": {
      "command": "python",
      "args": ["/path/to/yt-llm-service/simple-mcp-server/run.py"]
    }
  }
}
```

### Windows
Edit `%APPDATA%/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "yt-transcription": {
      "command": "python",
      "args": ["C:\\path\\to\\yt-llm-service\\simple-mcp-server\\run.py"]
    }
  }
}
```

### Linux
Edit `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "yt-transcription": {
      "command": "python",
      "args": ["/home/username/path/to/yt-llm-service/simple-mcp-server/run.py"]
    }
  }
}
```

## 🛠 Available Tools

Once configured, Claude will have access to these tools:

### `transcribe_youtube`
Transcribe YouTube videos with advanced speaker diarization.

**Parameters:**
- `youtube_url` (required): YouTube URL to transcribe
- `output_format`: simple, speaker, structured, markdown (default: simple)
- `min_speakers`: Minimum number of speakers (optional)
- `max_speakers`: Maximum number of speakers (optional)
- `remove_filler_words`: Remove "um", "uh" etc. (default: false)
- `merge_consecutive_speakers`: Merge consecutive segments (default: true)
- `verbose`: Enable detailed logging (default: true)

**Example Usage with Claude:**
> "Please transcribe this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ with speaker separation"

### `transcribe_file`
Transcribe local audio/video files.

**Parameters:**
- `audio_file_path` (required): Path to local audio/video file
- All other parameters same as `transcribe_youtube`

**Example Usage with Claude:**
> "Transcribe the file at /path/to/audio.mp3 in markdown format"

### `get_health`
Check service status and configuration.

**Example Usage with Claude:**
> "Check the health status of the transcription service"

## 📁 Available Resources

### `transcriptions://list`
List all saved transcriptions from the output directory.

**Example Usage with Claude:**
> "Show me all available transcriptions"

## 🔄 FastAPI to MCP Migration Pattern

This wrapper demonstrates how to convert FastAPI endpoints to MCP tools:

### FastAPI Endpoint
```python
@app.post("/transcribe-youtube")
async def transcribe_youtube(request: YouTubeRequest):
    # Implementation
    return response
```

### MCP Tool
```python
@mcp.tool()
async def transcribe_youtube(
    youtube_url: str = Field(description="YouTube URL"),
    # ... other parameters with Field descriptions
):
    # Same implementation, return dict instead of Pydantic model
    return response_dict
```

### Key Differences
1. **Decorators**: `@app.post()` → `@mcp.tool()`
2. **Parameters**: Request models → Individual parameters with `Field()` descriptions
3. **Responses**: Pydantic models → Plain dictionaries
4. **Transport**: HTTP → stdio
5. **Discovery**: OpenAPI docs → MCP protocol discovery

## 🏗 Project Structure

```
simple-mcp-server/
├── mcp_server.py      # Main MCP wrapper (imports from ../src/)
├── requirements.txt   # Minimal MCP dependencies
├── run.py            # Server runner script
└── README.md         # This file
```

## 🐛 Troubleshooting

### "ModuleNotFoundError" when running
- Ensure you're in the `simple-mcp-server` directory
- Check that the parent service dependencies are installed
- Verify the `sys.path.insert(0, '..')` line in `mcp_server.py`

### Claude Desktop not detecting tools
- Restart Claude Desktop after config changes
- Check the file path in your configuration is absolute
- Ensure Python is in your system PATH

### GPU/CUDA Issues
- The MCP server inherits GPU configuration from the parent service
- Check `../src/config.py` and `.env` files in the parent directory
- Verify CUDA setup with the parent FastAPI service first

## 🔧 Development

### Testing the Server
```bash
# Test the server directly
python run.py

# In another terminal, you can test with MCP client libraries
# or use it through Claude Desktop
```

### Adding New Tools
1. Add new `@mcp.tool()` functions to `mcp_server.py`
2. Import additional services from `../src/` as needed
3. The parent service handles all heavy lifting

### Environment Variables
The MCP server inherits all configuration from the parent service via `../src/config.py`. No additional environment setup needed.

## 📚 Learn More

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [Claude Desktop MCP Guide](https://docs.anthropic.com/en/docs/build-with-claude/mcp)

## 🎉 Benefits of This Approach

1. **No Code Duplication**: All business logic stays in one place
2. **Easy Maintenance**: Changes to services automatically reflected in MCP
3. **Dual Access**: Keep FastAPI for web clients, add MCP for LLMs
4. **Minimal Overhead**: Just a thin protocol wrapper
5. **Gradual Migration**: Can run both servers simultaneously