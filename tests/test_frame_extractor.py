"""Tests for frame_extractor module."""

from unittest.mock import patch, MagicMock

import pytest  # ty: ignore[unresolved-import]
from frame_extractor import extract_scene_frames, cleanup_frames, ExtractedFrame  # ty: ignore[unresolved-import]


def test_extract_scene_frames_calls_ffmpeg(tmp_path):
    """Verify ffmpeg is called with correct scene detection filter."""
    video = tmp_path / "test.mp4"
    video.touch()

    fake_stderr = (
        "[Parsed_showinfo_1 ...] pts_time:1.500\n"
        "[Parsed_showinfo_1 ...] pts_time:5.200\n"
    )

    # Create fake frame files that ffmpeg would produce
    with patch("frame_extractor.tempfile.mkdtemp") as mock_mkdtemp:
        mock_mkdtemp.return_value = str(tmp_path / "frames")
        (tmp_path / "frames").mkdir()
        (tmp_path / "frames" / "frame_0001.png").touch()
        (tmp_path / "frames" / "frame_0002.png").touch()

        with patch("frame_extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stderr=fake_stderr, returncode=0
            )
            frames = extract_scene_frames(str(video), scene_threshold=0.3, max_frames=50)

    assert len(frames) == 2
    assert frames[0].timestamp_s == 1.5
    assert frames[1].timestamp_s == 5.2
    assert frames[0].index == 0
    assert frames[1].index == 1

    # Verify ffmpeg was called with scene filter
    call_args = mock_run.call_args[0][0]
    assert "ffmpeg" in call_args[0]
    assert "select='gt(scene\\,0.3)'" in " ".join(call_args)


def test_extract_scene_frames_file_not_found():
    """Raise FileNotFoundError for missing video."""
    with pytest.raises(FileNotFoundError):
        extract_scene_frames("/nonexistent/video.mp4")


def test_cleanup_frames(tmp_path):
    """Verify cleanup deletes frames and parent dir."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    f1 = frame_dir / "frame_0001.png"
    f2 = frame_dir / "frame_0002.png"
    f1.touch()
    f2.touch()

    frames = [
        ExtractedFrame(path=str(f1), timestamp_s=1.0, index=0),
        ExtractedFrame(path=str(f2), timestamp_s=2.0, index=1),
    ]
    cleanup_frames(frames)

    assert not f1.exists()
    assert not f2.exists()
    assert not frame_dir.exists()


def test_cleanup_frames_empty_list():
    """cleanup_frames with empty list should not raise."""
    cleanup_frames([])
