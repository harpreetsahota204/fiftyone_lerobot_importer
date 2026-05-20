"""
Video splitter for extracting episode clips from sharded MP4s.

LeRobot v3.0 concatenates many episodes into a single MP4 per camera. This
module extracts a single episode's segment using the fastest available path:

- If the source is already H.264/yuv420p, the stream is copied (no decode).
- Otherwise the clip is re-encoded to H.264/yuv420p so the FiftyOne App
  (and browsers in general) can play it back.
"""

import subprocess
from pathlib import Path
from typing import Dict, Tuple, Union

try:
    import ffmpeg
except ImportError:
    raise ImportError(
        "ffmpeg-python is required for video splitting. "
        "Install it with: pip install ffmpeg-python"
    )


# Codec names that ffprobe may report for H.264 streams
_H264_CODECS = frozenset({"h264", "avc1", "avc"})

# Shared output flag: enable progressive download / faststart for streaming playback
_FASTSTART = "+faststart"


class VideoSplitter:
    """Extract per-episode MP4 clips from LeRobot's sharded source videos.
    
    Args:
        output_dir: Directory under which ``episode_{NNNNNN}/{camera}.mp4``
            clips will be written.
        overwrite: If True, re-extract clips even when the output already
            exists; if False (default), skip existing files.
    
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
        # codec/pix_fmt info is cached per source file because the same
        # sharded MP4 is used to extract many episodes
        self._codec_cache: Dict[str, bool] = {}
        
        self._verify_ffmpeg()
    
    @staticmethod
    def _verify_ffmpeg():
        """Fail fast with an actionable message if ffmpeg is missing."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH. "
                "Please install ffmpeg:\n"
                "  Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/download.html"
            ) from e
    
    def is_browser_compatible(self, video_path: Union[str, Path]) -> bool:
        """Return True if ``video_path`` is H.264/yuv420p and can be stream-copied.
        
        Results are cached per source file. If probing fails, the source is
        treated as incompatible so the caller falls back to re-encoding.
        """
        key = str(video_path)
        cached = self._codec_cache.get(key)
        if cached is not None:
            return cached
        
        try:
            codec, pix_fmt = probe_codec(key)
        except (RuntimeError, ValueError, FileNotFoundError, ffmpeg.Error):
            self._codec_cache[key] = False
            return False
        
        compatible = codec.lower() in _H264_CODECS and pix_fmt == "yuv420p"
        self._codec_cache[key] = compatible
        return compatible
    
    def split_episode(
        self,
        source_video: Union[str, Path],
        episode_index: int,
        camera_name: str,
        from_timestamp: float,
        to_timestamp: float,
    ) -> Path:
        """Extract one episode segment from a sharded source MP4.
        
        Uses stream copy when the source is already H.264/yuv420p; otherwise
        re-encodes to libx264 + yuv420p for browser-compatible playback.
        
        Args:
            source_video: Path to the sharded source MP4.
            episode_index: Episode number (used in output naming).
            camera_name: Camera name (used in output naming).
            from_timestamp: Start time in seconds within the source.
            to_timestamp: End time in seconds within the source.
        
        Returns:
            Path to the extracted clip.
        
        Raises:
            FileNotFoundError: If ``source_video`` doesn't exist.
            ValueError: If the timestamp range is non-positive.
            RuntimeError: If ffmpeg extraction fails.
        """
        source_video = Path(source_video)
        if not source_video.exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")
        
        duration = to_timestamp - from_timestamp
        if duration <= 0:
            raise ValueError(
                f"Invalid timestamps: from={from_timestamp}, to={to_timestamp}"
            )
        
        # Output layout: {output_dir}/episode_{NNNNNN}/{camera}.mp4
        episode_dir = self.output_dir / f"episode_{episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        output_path = episode_dir / f"{camera_name}.mp4"
        
        if output_path.exists() and not self.overwrite:
            return output_path
        
        if self.is_browser_compatible(source_video):
            # Stream copy: no decode/encode, just slice bytes out of the source
            output_kwargs = {"c": "copy", "movflags": _FASTSTART}
        else:
            # Re-encode to H.264/yuv420p; crf=23 + medium preset are libx264 defaults
            output_kwargs = {
                "vcodec": "libx264",
                "pix_fmt": "yuv420p",
                "crf": 23,
                "preset": "medium",
                "movflags": _FASTSTART,
            }
        
        try:
            (
                ffmpeg
                .input(str(source_video), ss=from_timestamp, t=duration)
                .output(str(output_path), **output_kwargs)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else "Unknown error"
            raise RuntimeError(
                f"Failed to extract episode {episode_index} camera "
                f"{camera_name} from {source_video}:\n{stderr}"
            ) from e
        
        return output_path


def probe_codec(video_path: Union[str, Path]) -> Tuple[str, str]:
    """Probe ``video_path`` and return ``(codec_name, pix_fmt)`` for the video stream.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        RuntimeError: If ffprobe fails.
        ValueError: If no video stream is found.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    try:
        probe = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to probe video {video_path}: {e}") from e
    
    stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    if stream is None:
        raise ValueError(f"No video stream found in {video_path}")
    
    return stream.get("codec_name", "unknown"), stream.get("pix_fmt", "unknown")
