import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import cli.py
sys.path.insert(0, str(Path(__file__).parent))
