#!/usr/bin/env python3
"""
Simple runner script for the FastMCP YouTube transcription server.

Usage:
    python run.py

This script starts the MCP server using stdio transport, which is the standard
way MCP servers communicate with clients like Claude Desktop.
"""

import asyncio
import sys
import os
from pathlib import Path

# Ensure we can import from the parent directory and current directory
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
# Make sure /app is in path for src imports
sys.path.insert(0, str(parent_dir))
# Make sure current dir is in path for mcp_server import
sys.path.insert(0, str(current_dir))

try:
    from mcp_server import mcp
except ImportError as e:
    print(f"Error importing MCP server: {e}", file=sys.stderr)
    print("Make sure you're in the simple-mcp-server directory and have installed dependencies.", file=sys.stderr)
    sys.exit(1)


def main():
    """Run the MCP server."""
    try:
        print("Starting YouTube Transcription MCP Server...", file=sys.stderr)
        print("Server will communicate via stdio. Press Ctrl+C to stop.", file=sys.stderr)

        # Run the FastMCP server (it handles the async loop internally)
        mcp.run()

    except KeyboardInterrupt:
        print("\nShutting down MCP server...", file=sys.stderr)
    except Exception as e:
        print(f"Error running MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Set up environment for running from this directory
    os.chdir(current_dir)

    # Run the server
    main()