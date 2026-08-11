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
        
        # Where OUTPUT_DIR is mounted from on the host. Set by docker-compose.
        # The service only ever sees container paths, but anything it writes
        # into an Obsidian vault is read on the host — see to_host_path().
        self.HOST_OUTPUT_DIR = os.getenv("HOST_OUTPUT_DIR")

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

        # Validate LLM output format
        valid_formats = ["simple", "speaker", "structured", "markdown"]
        if self.LLM_OUTPUT_FORMAT not in valid_formats:
            self.LLM_OUTPUT_FORMAT = "simple"

        # Ensure directories exist
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def to_host_path(self, container_path: Optional[str]) -> Optional[str]:
        """
        Rewrite a path under OUTPUT_DIR to the equivalent path on the host.

        Notes exported to an Obsidian vault are read by a human on the host,
        where the container's /app/output does not exist. Without this the
        recorded provenance points at nothing.

        Returns the path unchanged when no host mapping is configured or the
        path lies outside OUTPUT_DIR — never guesses.
        """
        if not container_path or not self.HOST_OUTPUT_DIR:
            return container_path

        container_root = str(self.OUTPUT_DIR).rstrip("/")
        if container_path == container_root:
            return self.HOST_OUTPUT_DIR.rstrip("/")
        if not container_path.startswith(container_root + "/"):
            return container_path

        relative = container_path[len(container_root) + 1:]
        return f"{self.HOST_OUTPUT_DIR.rstrip('/')}/{relative}"

    def __str__(self) -> str:
        """String representation hiding sensitive data"""
        return f"Config(DEVICE={self.DEVICE}, WHISPER_MODEL={self.WHISPER_MODEL}, BATCH_SIZE={self.BATCH_SIZE}, LLM_OUTPUT_FORMAT={self.LLM_OUTPUT_FORMAT}, LLAMA_CPP_URL={self.LLAMA_CPP_URL})"