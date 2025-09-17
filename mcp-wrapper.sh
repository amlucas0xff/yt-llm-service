#!/bin/bash
# MCP wrapper script for Claude Desktop
# Bridges stdio communication between Claude Desktop and the containerized MCP server

# Check if container is running
if ! docker ps --format "{{.Names}}" | grep -q "^yt-llm-service$"; then
    echo "Error: yt-llm-service container is not running" >&2
    echo "Please start it with: docker-compose up -d" >&2
    exit 1
fi

# Execute MCP server in container with stdio forwarding
exec docker exec -i yt-llm-service /app/entrypoint.sh mcp