"""Configuration management for the LLM Transcription Service"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """Simplified configuration class for LLM transcription service"""
    
    def __init__(self, env_file: Optional[str] = None):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv(env_file) if env_file else load_dotenv()
        
        # Core transcription settings
        self.DEVICE = os.getenv("DEVICE", "cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu")
        self.COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
        
        # Authentication
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        
        # Directories
        self.TEMP_DIR = Path(os.getenv("TEMP_DIR", "/app/tmp"))
        self.OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Ensure directories exist
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation hiding sensitive data"""
        return f"Config(DEVICE={self.DEVICE}, WHISPER_MODEL={self.WHISPER_MODEL}, BATCH_SIZE={self.BATCH_SIZE})"