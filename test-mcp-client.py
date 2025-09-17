#!/usr/bin/env python3
"""
MCP Test Client for YouTube Transcription Service

This script tests the MCP server functionality by sending JSON-RPC requests
and verifying responses. It helps validate that the MCP server is working
correctly before integrating with Claude Desktop.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

def send_mcp_request(request, timeout=30):
    """Send request to MCP server and get response"""
    try:
        proc = subprocess.Popen(
            ['./mcp-wrapper.sh'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent
        )

        # Send request with newline delimiter
        request_str = json.dumps(request) + '\n'
        stdout, stderr = proc.communicate(input=request_str, timeout=timeout)

        if proc.returncode != 0:
            print(f"Error: Process exited with code {proc.returncode}")
            if stderr:
                print(f"Stderr: {stderr}")
            return None

        if stderr:
            print(f"Server logs: {stderr}")

        # Try to parse response (handle case where multiple JSON objects might be present)
        if stdout.strip():
            try:
                # Split lines and find valid JSON responses
                lines = stdout.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            continue

                # If no valid JSON found, try parsing the whole stdout
                return json.loads(stdout.strip())
            except json.JSONDecodeError:
                print(f"Invalid JSON response: {stdout}")
                return None
        else:
            print("No response received")
            return None

    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"Request timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error executing request: {e}")
        return None

def test_mcp_server():
    """Run comprehensive tests of the MCP server"""
    print("=" * 60)
    print("MCP Server Test Suite for YouTube Transcription Service")
    print("=" * 60)

    # Test 1: Initialize MCP connection
    print("\n1. Testing MCP initialization...")
    init_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        },
        "id": 1
    }

    response = send_mcp_request(init_request)
    if response:
        print("✅ MCP initialization successful")
        print(f"Response: {json.dumps(response, indent=2)}")
    else:
        print("❌ MCP initialization failed")
        return False

    # Test 2: List available tools
    print("\n2. Testing tools/list...")
    list_tools_request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2
    }

    response = send_mcp_request(list_tools_request)
    if response:
        print("✅ Tools list successful")
        tools = response.get("result", {}).get("tools", [])
        print(f"Available tools: {[tool['name'] for tool in tools]}")
    else:
        print("❌ Tools list failed")
        return False

    # Test 3: Health check
    print("\n3. Testing get_health tool...")
    health_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "get_health",
            "arguments": {}
        },
        "id": 3
    }

    response = send_mcp_request(health_request)
    if response:
        print("✅ Health check successful")
        print(f"Health status: {json.dumps(response, indent=2)}")
    else:
        print("❌ Health check failed")
        return False

    # Test 4: List resources
    print("\n4. Testing resources/list...")
    list_resources_request = {
        "jsonrpc": "2.0",
        "method": "resources/list",
        "params": {},
        "id": 4
    }

    response = send_mcp_request(list_resources_request)
    if response:
        print("✅ Resources list successful")
        resources = response.get("result", {}).get("resources", [])
        print(f"Available resources: {[res['uri'] for res in resources]}")
    else:
        print("❌ Resources list failed")

    print("\n" + "=" * 60)
    print("✅ MCP Server testing completed successfully!")
    print("The server is ready for Claude Desktop integration.")
    print("=" * 60)
    return True

def main():
    """Main test runner"""
    # Check if wrapper script exists
    wrapper_path = Path(__file__).parent / "mcp-wrapper.sh"
    if not wrapper_path.exists():
        print("❌ Error: mcp-wrapper.sh not found")
        print("Please run this script from the yt-llm-service directory")
        sys.exit(1)

    # Check if wrapper is executable
    if not wrapper_path.stat().st_mode & 0o111:
        print("❌ Error: mcp-wrapper.sh is not executable")
        print("Run: chmod +x mcp-wrapper.sh")
        sys.exit(1)

    # Run tests
    success = test_mcp_server()

    if success:
        print("\n📋 Next steps:")
        print("1. Add to Claude Desktop MCP configuration:")
        print(f'   "yt-transcription": {{"command": "{wrapper_path.absolute()}"}}')
        print("2. Restart Claude Desktop")
        print("3. Test transcription tools in Claude Desktop")
    else:
        print("\n❌ Tests failed. Check the container logs:")
        print("   docker logs yt-llm-service")
        sys.exit(1)

if __name__ == "__main__":
    main()