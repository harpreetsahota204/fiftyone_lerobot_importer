"""
Video splitter for extracting episode clips from sharded MP4s.

This module provides utilities to split LeRobot v3.0 sharded video files
into individual episode clips for use with FiftyOne.

Clips are extracted using the fastest method available:
- If source is already H.264/yuv420p: uses stream copy (very fast)
- Otherwise: re-encodes to H.264/yuv420p for FiftyOne App compatibility
"""

import subprocess
from pathlib import Path
from typing import Union

try:
    import ffmpeg
except ImportError:
    raise ImportError(
        "ffmpeg-python is required for video splitting. "
        "Install it with: pip install ffmpeg-python"
    )


class VideoSplitter:
    """
    Splits sharded MP4 videos into individual episode clips.
    
    LeRobot v3.0 stores multiple episodes concatenated in single video files.
    This class extracts individual episode segments based on timestamps.
    
    Clips are extracted using the fastest method available:
    - If source is H.264/yuv420p: uses stream copy (very fast, no re-encoding)
    - Otherwise: re-encodes to H.264/yuv420p for FiftyOne App compatibility
    
    Args:
        output_dir: Directory to save extracted episode clips
        overwrite: If True, overwrite existing clips. If False, skip existing files.
    
    Example:
        >>> splitter = VideoSplitter("/path/to/clips")
        >>> clip_path = splitter.split_episode(
        ...     source_video="/path/to/sharded.mp4",
        ...     episode_index=0,
        ...     camera_name="cam_high",
        ...     from_timestamp=0.0,
        ...     to_timestamp=10.5,
        ... )
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        overwrite: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        
        # Cache codec info per source file (same sharded video used for many episodes)
        self._codec_cache: dict = {}
        
        # Verify ffmpeg is available
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self):
        """Verify that ffmpeg is installed and accessible."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH. "
                "Please install ffmpeg:\n"
                "  Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/download.html"
            )
    
    def _is_browser_compatible(self, video_path: Union[str, Path]) -> bool:
        """
        Check if video is already H.264/yuv420p (no re-encoding needed).
        
        Results are cached per source file since the same sharded video
        is used for many episode extractions.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video can be stream-copied without re-encoding
        """
        video_path = str(video_path)
        
        if video_path not in self._codec_cache:
            try:
                info = get_video_info(video_path)
                # H.264 codec names vary: h264, avc1, avc
                is_h264 = info["codec"].lower() in ("h264", "avc1", "avc")
                is_yuv420p = info["pix_fmt"] == "yuv420p"
                self._codec_cache[video_path] = is_h264 and is_yuv420p
            except Exception:
                # If we can't probe, assume we need to re-encode
                self._codec_cache[video_path] = False
        
        return self._codec_cache[video_path]
    
    def split_episode(
        self,
        source_video: Union[str, Path],
        episode_index: int,
        camera_name: str,
        from_timestamp: float,
        to_timestamp: float,
    ) -> Path:
        """
        Extract an episode clip from a sharded video file.
        
        Uses stream copy if source is already H.264/yuv420p (fast),
        otherwise re-encodes for browser compatibility.
        
        Args:
            source_video: Path to the sharded MP4 file
            episode_index: Episode index (used for output naming)
            camera_name: Camera name (used for output naming)
            from_timestamp: Start timestamp in seconds within the source video
            to_timestamp: End timestamp in seconds within the source video
            
        Returns:
            Path to the extracted episode clip
            
        Raises:
            FileNotFoundError: If source video doesn't exist
            RuntimeError: If ffmpeg extraction fails
        """
        source_video = Path(source_video)
        
        if not source_video.exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")
        
        # Create output path: output_dir/episode_NNNNNN/camera_name.mp4
        episode_dir = self.output_dir / f"episode_{episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        output_path = episode_dir / f"{camera_name}.mp4"
        
        # Skip if already extracted (unless overwrite is True)
        if output_path.exists() and not self.overwrite:
            return output_path
        
        # Calculate duration
        duration = to_timestamp - from_timestamp
        
        if duration <= 0:
            raise ValueError(
                f"Invalid timestamps: from={from_timestamp}, to={to_timestamp}"
            )
        
        try:
            # Check if source is already browser-compatible
            if self._is_browser_compatible(source_video):
                # Stream copy - FAST (no decode/encode, just copy bytes)
                (
                    ffmpeg
                    .input(str(source_video), ss=from_timestamp, t=duration)
                    .output(
                        str(output_path),
                        c="copy",              # Copy streams without re-encoding
                        movflags="+faststart", # Enable streaming playback
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                # Re-encode to H.264/yuv420p for browser compatibility
                (
                    ffmpeg
                    .input(str(source_video), ss=from_timestamp, t=duration)
                    .output(
                        str(output_path),
                        vcodec="libx264",      # H.264 codec
                        pix_fmt="yuv420p",     # Browser-compatible pixel format
                        crf=23,                # Quality (lower = better, 23 is default)
                        preset="medium",       # Encoding speed/compression tradeoff
                        movflags="+faststart", # Enable streaming playback
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
        except ffmpeg.Error as e:
            # Get stderr for debugging
            stderr = e.stderr.decode() if e.stderr else "Unknown error"
            raise RuntimeError(
                f"Failed to extract episode {episode_index} camera {camera_name} "
                f"from {source_video}:\n{stderr}"
            )
        
        return output_path
    
    def get_episode_clip_path(
        self,
        episode_index: int,
        camera_name: str,
    ) -> Path:
        """
        Get the expected path for an episode clip.
        
        Note: The file may not exist yet if split_episode() hasn't been called.
        
        Args:
            episode_index: Episode index
            camera_name: Camera name
            
        Returns:
            Expected path to the episode clip
        """
        return self.output_dir / f"episode_{episode_index:06d}" / f"{camera_name}.mp4"
    
    def episode_exists(self, episode_index: int, camera_name: str) -> bool:
        """Check if an episode clip has already been extracted."""
        return self.get_episode_clip_path(episode_index, camera_name).exists()


def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Get video metadata using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dict with keys: 'width', 'height', 'fps', 'duration', 'codec', 'pix_fmt'
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    try:
        probe = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to probe video {video_path}: {e}")
    
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"),
        None
    )
    
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")
    
    # Parse FPS (may be in format "30/1" or "29.97")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = map(float, fps_str.split("/"))
        fps = num / den if den != 0 else 30.0
    else:
        fps = float(fps_str)
    
    return {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "duration": float(probe.get("format", {}).get("duration", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
        "pix_fmt": video_stream.get("pix_fmt", "unknown"),
    }
