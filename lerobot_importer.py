"""
LeRobot v3.0 Dataset Importer for FiftyOne.

This module provides a FiftyOne dataset importer for LeRobot v3.0 format
robotics datasets. It creates grouped video samples where each group
represents an episode and each slice represents a camera view.

Frame-level data (observation states, actions) is stored using FiftyOne's
native video frame support.

Reference: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterator, Tuple

import pandas as pd
import numpy as np

import fiftyone as fo
import fiftyone.core.fields as fof
import fiftyone.core.groups as fog
import fiftyone.core.metadata as fom
import fiftyone.core.utils as focu
import fiftyone.types as fot
from fiftyone.utils.data.importers import GroupDatasetImporter

from video_splitter import VideoSplitter


class LeRobotDatasetImporter(GroupDatasetImporter):
    """
    Importer for LeRobot v3.0 format robotics datasets.
    
    Creates grouped video samples where:
    - Each group = one episode
    - Each slice = one camera view (video file)
    - Frame-level data (states, actions) stored in sample.frames
    
    Supports fo.Dataset.from_dir() pattern:
    ```python
    dataset = fo.Dataset.from_dir(
        dataset_dir="/path/to/dataset",
        dataset_type=LeRobotDataset,
        camera_views=["cam_high", "cam_low"],
    )
    ```
    """
    
    def __init__(
        self,
        dataset_dir: Union[str, Path] = None,
        data_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        camera_views: Optional[List[str]] = None,
        episode_ids: Optional[List[int]] = None,
        task_ids: Optional[List[int]] = None,
        clips_dir: Optional[Union[str, Path]] = None,
        include_frame_data: bool = True,
        include_metadata: bool = True,  # Alias for include_frame_data
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        default_slice: Optional[str] = None,
        group_field: str = "group",
        overwrite_clips: bool = False,
        **kwargs,
    ):
        """
        Initialize the importer.
        
        Args:
            dataset_dir: Root directory of the v3.0 dataset
            data_path: Not used (for API compatibility)
            labels_path: Not used (for API compatibility) 
            camera_views: List of camera views to import (None = auto-detect)
            episode_ids: Specific episode IDs to import (None = all)
            task_ids: Filter by task IDs (None = all)
            clips_dir: Directory for extracted episode clips
            include_frame_data: Whether to load frame-level states/actions
            include_metadata: Alias for include_frame_data
            max_samples: Maximum number of episodes to import
            shuffle: Whether to shuffle episodes
            seed: Random seed for shuffling
            default_slice: Default camera slice name
            group_field: Name of the group field
            overwrite_clips: Whether to overwrite existing clips
        """
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.camera_views = camera_views
        self.episode_ids = episode_ids
        self.task_ids = task_ids
        self.clips_dir = Path(clips_dir) if clips_dir else None
        self.include_frame_data = include_frame_data and include_metadata
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        self.default_slice = default_slice
        self._group_field = group_field
        self.overwrite_clips = overwrite_clips
        
        # Internal state
        self._dataset_info: Optional[Dict] = None
        self._episodes_df: Optional[pd.DataFrame] = None
        self._tasks_df: Optional[pd.DataFrame] = None
        self._stats: Dict = {}  # stats.json contents
        self._task_mapping: Dict[str, int] = {}  # task_string -> task_index
        self._data_df_cache: Dict[tuple, pd.DataFrame] = {}
        self._video_splitter: Optional[VideoSplitter] = None
        self._episodes_to_import: Optional[List[Dict]] = None
        self._fps: int = 30
        
        # Video feature mapping: short name -> full feature name
        # e.g., {"top": "observation.images.top", "wrist": "observation.images.wrist"}
        self._video_feature_map: Dict[str, str] = {}
        
        # Iteration state
        self._samples_iter: Optional[Iterator] = None
        self._num_samples: int = 0
    
    @property
    def group_field(self) -> str:
        """The name of the group field."""
        return self._group_field
    
    @property
    def has_dataset_info(self) -> bool:
        """Whether this importer produces dataset info."""
        return True
    
    @property
    def has_video_metadata(self) -> bool:
        """Whether this importer produces video metadata."""
        return True
    
    @property
    def has_sample_field_schema(self) -> bool:
        """Whether this importer produces a sample field schema."""
        return False
    
    def setup(self):
        """Setup method called before iteration begins."""
        if self.dataset_dir is None:
            raise ValueError("dataset_dir is required")
        
        # Set clips_dir default
        if self.clips_dir is None:
            self.clips_dir = self.dataset_dir / "episode_clips"
        
        # Validate v3.0 structure
        self._validate_v3_structure()
        
        # Load metadata
        self._load_metadata()
        
        # Initialize video splitter
        self._video_splitter = VideoSplitter(
            self.clips_dir, 
            overwrite=self.overwrite_clips
        )
        
        # Build list of episodes to import
        self._build_episodes_list()
        
        # Pre-compute all samples for iteration
        self._build_samples_iterator()
        
        print(f"Setup complete: {self._num_samples} samples from "
              f"{len(self._episodes_to_import)} episodes")
    
    def _validate_v3_structure(self):
        """Validate that dataset is v3.0 format."""
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")
        
        info_path = self.dataset_dir / "meta" / "info.json"
        if not info_path.exists():
            raise ValueError(
                f"Not a valid LeRobot dataset: {info_path} not found. "
                f"Expected v3.0 format with meta/info.json"
            )
        
        with open(info_path) as f:
            info = json.load(f)
        
        version = info.get("codebase_version", "")
        if not version.startswith("v3"):
            raise ValueError(
                f"This importer only supports LeRobot v3.0 format. "
                f"Found version: '{version}'. "
                f"Please convert your dataset using LeRobot's conversion tools."
            )
        
        # Check for required directories
        required_dirs = [
            ("data", "Parquet data files"),
            ("videos", "Video files"),
            ("meta/episodes", "Episode metadata"),
        ]
        
        for dir_path, description in required_dirs:
            full_path = self.dataset_dir / dir_path
            if not full_path.exists():
                raise ValueError(
                    f"Missing required directory: {full_path} ({description})"
                )
        
        print(f"Validated v3.0 dataset at {self.dataset_dir}")
    
    def _load_metadata(self):
        """Load all metadata from the dataset."""
        meta_dir = self.dataset_dir / "meta"
        
        # Load info.json
        with open(meta_dir / "info.json") as f:
            self._dataset_info = json.load(f)
        
        self._fps = int(self._dataset_info.get("fps", 30))
        
        # Load stats.json (normalization statistics for ML training)
        stats_path = meta_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self._stats = json.load(f)
        else:
            self._stats = {}
        
        # Auto-detect camera views from features if not specified
        if self.camera_views is None:
            self.camera_views = self._detect_camera_views()
            print(f"Auto-detected camera views: {self.camera_views}")
        
        # Set default slice
        if self.default_slice is None and self.camera_views:
            self.default_slice = self.camera_views[0]
        
        # Load episodes metadata (chunked parquet)
        episodes_dir = meta_dir / "episodes"
        episode_files = sorted(episodes_dir.glob("**/*.parquet"))
        
        if not episode_files:
            raise ValueError(f"No episode metadata found in {episodes_dir}")
        
        self._episodes_df = pd.concat(
            [pd.read_parquet(f) for f in episode_files],
            ignore_index=True
        )
        
        # Load tasks metadata - support multiple formats per v3.0 evolution
        # 1. tasks.parquet (single file) - common in practice
        # 2. tasks.jsonl (single file) - per official docs
        # 3. tasks/ directory (chunked parquet) - for scalability
        tasks_parquet = meta_dir / "tasks.parquet"
        tasks_jsonl = meta_dir / "tasks.jsonl"
        tasks_dir = meta_dir / "tasks"
        
        self._task_mapping = {}  # task_string -> task_index
        
        if tasks_parquet.exists():
            # Single parquet file - task strings are the index
            self._tasks_df = pd.read_parquet(tasks_parquet)
            # Build mapping: index is task string, column is task_index
            for task_str, row in self._tasks_df.iterrows():
                self._task_mapping[task_str] = int(row["task_index"])
        elif tasks_jsonl.exists():
            # JSONL format per official v3.0 docs
            self._tasks_df = pd.read_json(tasks_jsonl, lines=True)
            # Assuming columns: task, task_index
            for _, row in self._tasks_df.iterrows():
                self._task_mapping[row.get("task", "")] = int(row.get("task_index", 0))
        elif tasks_dir.exists():
            # Chunked parquet directory
            task_files = sorted(tasks_dir.glob("**/*.parquet"))
            if task_files:
                self._tasks_df = pd.concat(
                    [pd.read_parquet(f) for f in task_files],
                    ignore_index=True
                )
                for _, row in self._tasks_df.iterrows():
                    self._task_mapping[row.get("task", "")] = int(row.get("task_index", 0))
        
        print(f"Loaded metadata: {len(self._episodes_df)} episodes, "
              f"{len(self.camera_views)} cameras, {len(self._task_mapping)} tasks, FPS={self._fps}")
    
    def _detect_camera_views(self) -> List[str]:
        """
        Auto-detect camera views from dataset features.
        
        Also builds _video_feature_map mapping short names to full feature names.
        e.g., {"top": "observation.images.top", "wrist": "observation.images.wrist"}
        """
        features = self._dataset_info.get("features", {})
        cameras = []
        
        for key, feat in features.items():
            if feat.get("dtype") == "video":
                # Extract short name from full feature name
                # e.g., "observation.images.top" -> "top"
                short_name = key.split(".")[-1]
                cameras.append(short_name)
                # Store mapping: short name -> full feature name
                self._video_feature_map[short_name] = key
        
        if not cameras:
            # Fallback: scan videos directory
            videos_dir = self.dataset_dir / "videos"
            if videos_dir.exists():
                for d in videos_dir.iterdir():
                    if d.is_dir() and not d.name.startswith("."):
                        # Directory name is the full feature name
                        full_name = d.name
                        short_name = full_name.split(".")[-1]
                        cameras.append(short_name)
                        self._video_feature_map[short_name] = full_name
        
        return sorted(cameras)
    
    def _build_episodes_list(self):
        """Build filtered list of episodes to import."""
        episodes = self._episodes_df.to_dict("records")
        
        if self.episode_ids is not None:
            episode_ids_set = set(self.episode_ids)
            episodes = [e for e in episodes if e["episode_index"] in episode_ids_set]
        
        if self.task_ids is not None:
            task_ids_set = set(self.task_ids)
            episodes = [e for e in episodes if e.get("task_index") in task_ids_set]
        
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(episodes)
        
        if self.max_samples is not None:
            episodes = episodes[:self.max_samples]
        
        self._episodes_to_import = episodes
    
    def _build_samples_iterator(self):
        """Build iterator over all sample groups (one group per episode)."""
        all_groups = []
        
        for episode in self._episodes_to_import:
            # Returns list of sample dicts for this episode (one per camera)
            samples = self._create_episode_samples(episode)
            if samples:
                all_groups.append(samples)
        
        # Count total individual samples
        self._num_samples = sum(len(group) for group in all_groups)
        self._samples_iter = iter(all_groups)
    
    def _resolve_video_path(self, camera: str, chunk_idx: int, file_idx: int) -> Path:
        """
        Resolve path to sharded video file.
        
        Uses the video_path template from info.json with the full feature name.
        """
        # Get full feature name from mapping (e.g., "top" -> "observation.images.top")
        video_key = self._video_feature_map.get(camera, camera)
        
        # Use path template from info.json if available
        path_template = self._dataset_info.get(
            "video_path", 
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        
        # Format the path
        rel_path = path_template.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        
        return self.dataset_dir / rel_path
    
    def _resolve_data_path(self, chunk_idx: int, file_idx: int) -> Path:
        """Resolve path to sharded parquet file using template from info.json."""
        path_template = self._dataset_info.get(
            "data_path",
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        )
        
        rel_path = path_template.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        
        return self.dataset_dir / rel_path
    
    def _load_episode_frame_data(self, episode: Dict) -> Optional[pd.DataFrame]:
        """Load frame-level data for an episode from parquet."""
        chunk_idx = episode.get("data/chunk_index", episode.get("data_chunk_index", 0))
        file_idx = episode.get("data/file_index", episode.get("data_file_index", 0))
        from_idx = episode.get("dataset_from_index", 0)
        to_idx = episode.get("dataset_to_index", from_idx + episode.get("length", 0))
        
        cache_key = (chunk_idx, file_idx)
        
        if cache_key not in self._data_df_cache:
            parquet_path = self._resolve_data_path(chunk_idx, file_idx)
            if parquet_path.exists():
                self._data_df_cache[cache_key] = pd.read_parquet(parquet_path)
            else:
                print(f"Warning: Parquet file not found: {parquet_path}")
                return None
        
        df = self._data_df_cache[cache_key]
        return df.iloc[from_idx:to_idx].reset_index(drop=True)
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a value for FiftyOne/MongoDB compatibility."""
        if value is None:
            return None
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        else:
            return value
    
    def _create_episode_samples(self, episode: Dict) -> List[Dict]:
        """Create sample dicts for one episode (all cameras)."""
        episode_idx = episode["episode_index"]
        group_id = focu.ObjectId()
        
        frame_data_df = None
        if self.include_frame_data:
            frame_data_df = self._load_episode_frame_data(episode)
        
        # Get episode-level metadata
        episode_length = int(episode.get("length", 0))
        dataset_from_index = episode.get("dataset_from_index")
        dataset_to_index = episode.get("dataset_to_index")
        
        # Get task info from episode's tasks list (v3.0 format)
        tasks_list = episode.get("tasks", [])
        task_string = tasks_list[0] if tasks_list else None
        task_index = self._task_mapping.get(task_string) if task_string else None
        
        # Get video dimensions from features (same for all cameras of same type)
        # Default to None, will be populated per-camera if available
        
        samples = []
        
        for camera in self.camera_views:
            # Get full feature name from mapping
            # e.g., "top" -> "observation.images.top"
            video_key = self._video_feature_map.get(camera, camera)
            
            # Episode metadata uses format: videos/{video_key}/chunk_index
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"
            from_key = f"videos/{video_key}/from_timestamp"
            to_key = f"videos/{video_key}/to_timestamp"
            
            if chunk_key not in episode:
                continue
            
            chunk_idx = int(episode[chunk_key])
            file_idx = int(episode[file_key])
            from_ts = float(episode[from_key])
            to_ts = float(episode[to_key])
            
            source_video = self._resolve_video_path(camera, chunk_idx, file_idx)
            
            if not source_video.exists():
                print(f"Warning: Video not found: {source_video}")
                continue
            
            try:
                clip_path = self._video_splitter.split_episode(
                    source_video=source_video,
                    episode_index=episode_idx,
                    camera_name=camera,
                    from_timestamp=from_ts,
                    to_timestamp=to_ts,
                )
            except Exception as e:
                print(f"Warning: Failed to extract episode {episode_idx} "
                      f"camera {camera}: {e}")
                continue
            
            # Get video dimensions from features for this camera
            features = self._dataset_info.get("features", {})
            camera_feature = features.get(video_key, {})
            shape = camera_feature.get("shape", [])  # e.g., [480, 640, 3]
            frame_height = shape[0] if len(shape) > 0 else None
            frame_width = shape[1] if len(shape) > 1 else None
            
            # Calculate duration from timestamps
            duration = to_ts - from_ts
            
            # Build VideoMetadata
            video_metadata = fom.VideoMetadata(
                frame_width=frame_width,
                frame_height=frame_height,
                frame_rate=float(self._fps),
                total_frame_count=episode_length,
                duration=duration,
                mime_type="video/mp4",
                encoding_str="avc1",  # H.264 (we re-encode to this)
            )
            
            # Build sample dict
            sample_dict = {
                "filepath": str(clip_path),
                "group_id": group_id,
                "group_name": camera,
                "metadata": video_metadata,
                # Episode-level fields
                "episode_index": episode_idx,
                "camera_view": camera,
                # Task info
                "task": task_string,
                "task_index": task_index,
                # Global dataset position
                "dataset_from_index": dataset_from_index,
                "dataset_to_index": dataset_to_index,
                # Frame data for later processing
                "frame_data": frame_data_df,
            }
            
            samples.append(sample_dict)
        
        return samples
    
    def _add_frame_data_to_sample(self, sample: fo.Sample, frame_data: pd.DataFrame):
        """Add frame-level data to video sample's frames."""
        for frame_idx, row in frame_data.iterrows():
            fo_frame_num = int(frame_idx) + 1
            frame = sample.frames[fo_frame_num]
            
            if "observation.state" in row.index:
                frame["observation_state"] = self._sanitize_value(row["observation.state"])
            
            if "action" in row.index:
                frame["action"] = self._sanitize_value(row["action"])
            
            if "timestamp" in row.index:
                frame["timestamp"] = float(row["timestamp"])
            
            if "index" in row.index:
                frame["global_index"] = int(row["index"])
    
    def __len__(self) -> int:
        """Return number of groups (episodes) to import."""
        if self._episodes_to_import is None:
            return 0
        return len(self._episodes_to_import)
    
    def __iter__(self):
        """Initialize iteration."""
        self._build_samples_iterator()
        return self
    
    def __next__(self) -> Dict[str, fo.Sample]:
        """
        Return the next group of samples (all camera views for one episode).
        
        Returns:
            Dict mapping slice names to fo.Sample instances
        """
        # Get all sample dicts for this episode (one per camera)
        sample_dicts = next(self._samples_iter)
        
        # Build dict mapping slice name -> fo.Sample
        group_samples = {}
        
        for sample_dict in sample_dicts:
            # Create the FiftyOne sample
            sample = fo.Sample(filepath=sample_dict["filepath"])
            
            # Set VideoMetadata
            sample.metadata = sample_dict["metadata"]
            
            # Episode identification
            sample["episode_index"] = sample_dict["episode_index"]
            sample["camera_view"] = sample_dict["camera_view"]
            
            # Task info
            if sample_dict.get("task") is not None:
                sample["task"] = sample_dict["task"]
            if sample_dict.get("task_index") is not None:
                sample["task_index"] = sample_dict["task_index"]
            
            # Global dataset position (for mapping back to original dataset)
            if sample_dict.get("dataset_from_index") is not None:
                sample["dataset_from_index"] = int(sample_dict["dataset_from_index"])
            if sample_dict.get("dataset_to_index") is not None:
                sample["dataset_to_index"] = int(sample_dict["dataset_to_index"])
            
            # Add group
            sample[self._group_field] = fog.Group(
                id=sample_dict["group_id"],
                name=sample_dict["group_name"]
            )
            
            # Add frame data
            if sample_dict.get("frame_data") is not None:
                self._add_frame_data_to_sample(sample, sample_dict["frame_data"])
            
            group_samples[sample_dict["group_name"]] = sample
        
        return group_samples
    
    def get_group_field(self) -> str:
        """Return the group field name."""
        return self._group_field
    
    def get_group_media_types(self) -> Dict[str, str]:
        """Return media types for each group slice."""
        return {camera: "video" for camera in self.camera_views}
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset info dict with full metadata for ML training."""
        if self._dataset_info is None:
            return {}
        
        return {
            # Dataset identification
            "type": "LeRobot v3.0 Dataset",
            "format": "grouped_video",
            "codebase_version": self._dataset_info.get("codebase_version", "v3.0"),
            "robot_type": self._dataset_info.get("robot_type"),
            
            # Episode counts
            "episode_count": len(self._episodes_to_import) if self._episodes_to_import else 0,
            "total_episodes": self._dataset_info.get("total_episodes", 0),
            "total_frames": self._dataset_info.get("total_frames", 0),
            
            # Structure
            "camera_views": self.camera_views,
            "default_slice": self.default_slice,
            "group_field": self._group_field,
            "fps": self._fps,
            
            # Feature definitions (shapes, dtypes) - important for understanding data
            "features": self._dataset_info.get("features", {}),
            
            # Normalization statistics - critical for ML training
            "stats": self._stats,
            
            # Task vocabulary
            "tasks": self._task_mapping,
        }
    
    def close(self, *args):
        """Clean up resources."""
        self._data_df_cache.clear()
        self._samples_iter = None


class LeRobotDataset(fot.Dataset):
    """
    Dataset type for LeRobot v3.0 robotics datasets.
    
    Use with fo.Dataset.from_dir():
    ```python
    dataset = fo.Dataset.from_dir(
        dataset_dir="/path/to/dataset",
        dataset_type=LeRobotDataset,
        camera_views=["cam_high", "cam_low"],
        name="my_dataset",
    )
    ```
    """
    
    def get_dataset_importer_cls(self):
        """Return the importer class for this dataset type."""
        return LeRobotDatasetImporter


# Convenience function for direct import
def import_lerobot_dataset(
    dataset_dir: Union[str, Path],
    name: Optional[str] = None,
    camera_views: Optional[List[str]] = None,
    episode_ids: Optional[List[int]] = None,
    task_ids: Optional[List[int]] = None,
    include_frame_data: bool = True,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    **kwargs,
) -> fo.Dataset:
    """
    Convenience function to import a LeRobot v3.0 dataset.
    
    Args:
        dataset_dir: Root directory of the v3.0 dataset
        name: Name for the FiftyOne dataset (default: derived from directory)
        camera_views: List of camera views to import (None = auto-detect)
        episode_ids: Specific episode IDs to import (None = all)
        task_ids: Filter by task IDs (None = all)
        include_frame_data: Whether to load frame-level states/actions
        max_samples: Maximum number of episodes to import
        overwrite: Whether to overwrite existing FiftyOne dataset
        **kwargs: Additional arguments passed to LeRobotDatasetImporter
        
    Returns:
        FiftyOne dataset with imported samples
    """
    dataset_dir = Path(dataset_dir)
    
    if name is None:
        name = dataset_dir.name
    
    if fo.dataset_exists(name):
        if overwrite:
            fo.delete_dataset(name)
        else:
            raise ValueError(
                f"Dataset '{name}' already exists. "
                f"Use overwrite=True to replace it."
            )
    
    # Use from_dir with our dataset type
    dataset = fo.Dataset.from_dir(
        dataset_dir=str(dataset_dir),
        dataset_type=LeRobotDataset,
        name=name,
        camera_views=camera_views,
        episode_ids=episode_ids,
        task_ids=task_ids,
        include_frame_data=include_frame_data,
        max_samples=max_samples,
        **kwargs,
    )
    
    return dataset


# Register the dataset type with FiftyOne
fot.LeRobotDataset = LeRobotDataset
