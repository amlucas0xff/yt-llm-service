"""Extract key frames from video using ffmpeg scene change detection."""

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    path: str
    timestamp_s: float
    index: int


def extract_scene_frames(
    video_path: str,
    scene_threshold: float = 0.3,
    max_frames: int = 100,
) -> list[ExtractedFrame]:
    """Extract frames at scene changes from a video file.

    Uses ffmpeg select filter with scene change detection.
    Returns list of ExtractedFrame sorted by timestamp.
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = tempfile.mkdtemp(prefix="ocr_frames_")
    output_pattern = str(Path(output_dir) / "frame_%04d.png")

    # ffmpeg scene detection with showinfo for timestamps
    cmd = [
        "ffmpeg", "-i", str(video),
        "-vf", f"select='gt(scene\\,{scene_threshold})',showinfo",
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        output_pattern,
        "-y",
    ]

    logger.info(f"Extracting scene frames: threshold={scene_threshold}, max={max_frames}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
    )

    # Parse showinfo output from stderr for PTS timestamps
    # Pattern: [Parsed_showinfo_1 ...] pts_time:123.456
    pts_pattern = re.compile(r"pts_time:\s*([\d.]+)")
    timestamps = pts_pattern.findall(result.stderr)

    # Collect extracted frames
    frames = []
    frame_dir = Path(output_dir)
    for idx, frame_path in enumerate(sorted(frame_dir.glob("frame_*.png"))):
        ts = float(timestamps[idx]) if idx < len(timestamps) else 0.0
        frames.append(ExtractedFrame(
            path=str(frame_path),
            timestamp_s=ts,
            index=idx,
        ))

    logger.info(f"Extracted {len(frames)} scene-change frames")
    return frames


def cleanup_frames(frames: list[ExtractedFrame]) -> None:
    """Delete extracted frame files and their parent temp directory."""
    if not frames:
        return
    parent = Path(frames[0].path).parent
    for f in frames:
        Path(f.path).unlink(missing_ok=True)
    parent.rmdir()
