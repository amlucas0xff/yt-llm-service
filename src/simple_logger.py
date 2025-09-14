"""
Simple human-readable logging utility for yt-llm-service
"""
import os
from datetime import datetime
from pathlib import Path
import inspect
from typing import Optional


class SimpleLogger:
    """Simple logger that outputs human-readable logs to a file"""
    
    def __init__(self, log_file: str = "logs/log.txt"):
        """
        Initialize the logger
        
        Args:
            log_file: Path to the log file (relative to project root)
        """
        # Get the project root (yt-llm-service directory)
        project_root = Path(__file__).parent.parent
        self.log_file = project_root / log_file
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create log file if it doesn't exist
        if not self.log_file.exists():
            self.log_file.touch()
    
    def log(self, action: str, extra_data: Optional[str] = None):
        """
        Log an action with caller information
        
        Args:
            action: Description of the action being performed
            extra_data: Optional additional data to include
        """
        try:
            # Get caller information
            frame = inspect.currentframe().f_back
            file_path = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            
            # Extract relative path from src/ or project root
            if "/src/" in file_path:
                relative_path = file_path.split("/src/")[1]
            else:
                # Try to get relative path from project root
                try:
                    project_root = Path(__file__).parent.parent
                    relative_path = str(Path(file_path).relative_to(project_root))
                except ValueError:
                    # If that fails, just use the filename
                    relative_path = Path(file_path).name
            
            # Format timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Build log entry
            log_parts = [timestamp, relative_path, f"{function_name}()", action]
            if extra_data:
                log_parts.append(f"[{extra_data}]")
            
            log_entry = " | ".join(log_parts) + "\n"
            
            # Write to log file
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
                
        except Exception as e:
            # If logging fails, write to stderr but don't raise
            import sys
            print(f"Logging error: {e}", file=sys.stderr)


# Global logger instance
logger = SimpleLogger()


def log_action(action: str, extra_data: Optional[str] = None):
    """
    Convenience function to log an action
    
    Args:
        action: Description of the action being performed
        extra_data: Optional additional data to include
    """
    logger.log(action, extra_data)