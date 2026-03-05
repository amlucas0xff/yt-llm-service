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

        # LLM-specific configuration options
        self.LLM_OUTPUT_FORMAT = os.getenv("LLM_OUTPUT_FORMAT", "simple")  # simple, speaker, structured, markdown
        self.LLM_REMOVE_FILLER_WORDS = os.getenv("LLM_REMOVE_FILLER_WORDS", "false").lower() == "true"
        self.LLM_MERGE_CONSECUTIVE_SPEAKERS = os.getenv("LLM_MERGE_CONSECUTIVE_SPEAKERS", "true").lower() == "true"

        # Notes service configuration
        self.LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://llama-cpp:8080")
        self.NOTES_MAX_TOKENS = int(os.getenv("NOTES_MAX_TOKENS", "80000"))
        # LLAMA_CPP_GPU_LAYERS is consumed by docker-compose, stored here for logging only
        self.LLAMA_CPP_GPU_LAYERS = int(os.getenv("LLAMA_CPP_GPU_LAYERS", "99"))

        # OCR service configuration
        self.OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://ocr-service:8003")
        self.OCR_SERVICE_TIMEOUT = float(os.getenv("OCR_SERVICE_TIMEOUT", "60"))
        self.OCR_SCENE_THRESHOLD = float(os.getenv("OCR_SCENE_THRESHOLD", "0.3"))
        self.OCR_MAX_FRAMES = int(os.getenv("OCR_MAX_FRAMES", "100"))

        # Validate LLM output format
        valid_formats = ["simple", "speaker", "structured", "markdown"]
        if self.LLM_OUTPUT_FORMAT not in valid_formats:
            self.LLM_OUTPUT_FORMAT = "simple"

        # Ensure directories exist
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation hiding sensitive data"""
        return f"Config(DEVICE={self.DEVICE}, WHISPER_MODEL={self.WHISPER_MODEL}, BATCH_SIZE={self.BATCH_SIZE}, LLM_OUTPUT_FORMAT={self.LLM_OUTPUT_FORMAT}, LLAMA_CPP_URL={self.LLAMA_CPP_URL})"