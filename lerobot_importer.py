"""
LeRobot v3.0 Dataset Importer for FiftyOne.

This module provides a FiftyOne dataset importer for LeRobot v3.0 format
robotics datasets. It creates grouped video samples where each group
represents an episode and each slice represents a camera view.

Frame-level data (observation states, actions) is stored using FiftyOne's
native video frame support.

Reference: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
"""

import fnmatch
import json
import math
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

import fiftyone as fo
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
        camera_views: Optional[List[str]] = None,
        episode_ids: Optional[List[int]] = None,
        task_ids: Optional[List[int]] = None,
        clips_dir: Optional[Union[str, Path]] = None,
        include_frame_data: bool = True,
        include_fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
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
            camera_views: List of camera views to import (None = auto-detect)
            episode_ids: Specific episode IDs to import (None = all)
            task_ids: Filter by task IDs (None = all)
            clips_dir: Directory for extracted episode clips
            include_frame_data: Whether to load frame-level states/actions
            include_fields: Glob patterns for LeRobot field names to include
                (e.g., ["observation.*", "action.*", "timestamp"]).
                None means include all non-video fields.
            exclude_fields: Glob patterns for LeRobot field names to exclude
                (e.g., ["*.is_fresh"]). Applied after include_fields.
                None means exclude nothing.
            max_samples: Maximum number of episodes to import
            shuffle: Whether to shuffle episodes
            seed: Random seed for shuffling
            default_slice: Default camera slice name
            group_field: Name of the group field
            overwrite_clips: Whether to overwrite existing clips
            **kwargs: Absorbs forwarded ``from_dir`` arguments (data_path,
                labels_path, etc.) that this importer does not consume.
        """
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.camera_views = camera_views
        self.episode_ids = episode_ids
        self.task_ids = task_ids
        self.clips_dir = Path(clips_dir) if clips_dir else None
        self.include_frame_data = include_frame_data
        self.include_fields = include_fields
        self.exclude_fields = exclude_fields
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        self.default_slice = default_slice
        self._group_field = group_field
        self.overwrite_clips = overwrite_clips
        
        # Loaded metadata
        self._dataset_info: Optional[Dict] = None
        self._features: Dict[str, Dict] = {}      # info.json["features"], hoisted
        self._stats: Dict = {}                     # stats.json contents
        self._fps: int = 30
        self._video_path_template: str = ""
        self._data_path_template: str = ""
        
        # Episode + task tables
        self._episodes: Optional[List[Dict]] = None
        self._episodes_to_import: Optional[List[Dict]] = None
        self._task_mapping: Dict[str, int] = {}    # task_string -> task_index
        self._tasks_by_index: Dict[int, str] = {}  # task_index -> task_string
        # Trusted per-episode data location, rebuilt from actual shard
        # contents at setup time. Some v3.0 datasets ship with stale
        # data/chunk_index / data/file_index in episode metadata, and their
        # dataset_from_index is a global row index that doesn't translate
        # to a within-shard offset, so we can't trust the episode metadata
        # alone. This map is the source of truth instead.
        #     episode_index -> ((chunk_idx, file_idx), from_row, to_row)
        # where (from_row, to_row) are within-shard slice offsets.
        self._episode_location: Dict[
            int, Tuple[Tuple[int, int], int, int]
        ] = {}
        
        # Per-camera derived state, computed once after metadata load:
        #   _video_feature_map[camera] -> full feature key
        #   _camera_episode_keys[camera] -> (chunk_key, file_key, from_key, to_key)
        #   _camera_dimensions[camera] -> (frame_height, frame_width) or (None, None)
        self._video_feature_map: Dict[str, str] = {}
        self._camera_episode_keys: Dict[str, Tuple[str, str, str, str]] = {}
        self._camera_dimensions: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        
        # Frame field schema, built from info.json features:
        #   _frame_fields[fo_name] = {"lerobot_name", "dtype", "shape"}
        #   _field_names_meta[fo_name] = [semantic names]
        #   _field_descriptions[fo_name] = human-readable string
        self._frame_fields: Dict[str, Dict] = {}
        self._field_names_meta: Dict[str, List[str]] = {}
        self._field_descriptions: Dict[str, str] = {}
        self._columns_to_read: List[str] = []
        
        # Runtime state
        self._video_splitter: Optional[VideoSplitter] = None
        self._data_table_cache: Dict[Tuple[int, int], Any] = {}
        # Separate light-weight cache for the per-frame task_index column,
        # used as a fallback when episode metadata lacks a 'tasks' field
        self._task_index_cache: Dict[Tuple[int, int], Any] = {}
        self._samples_iter: Optional[Iterator] = None
        # (chunk_idx, file_idx) -> last episode_index using that shard;
        # drives parquet cache eviction during lazy iteration
        self._last_episode_for_shard: Dict[Tuple[int, int], int] = {}
    
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
        """Prepare the importer for iteration.
        
        Performs lightweight work only — validation, metadata loading,
        episode filtering, and cache pre-computation. Clip extraction and
        parquet reads happen lazily during iteration so the first sample
        flows immediately rather than after the entire ffmpeg pass.
        """
        if self.dataset_dir is None:
            raise ValueError("dataset_dir is required")
        
        if self.clips_dir is None:
            self.clips_dir = self.dataset_dir / "episode_clips"
        
        info = self._validate_v3_structure()
        self._load_metadata(info)
        
        self._video_splitter = VideoSplitter(
            self.clips_dir, overwrite=self.overwrite_clips
        )
        
        self._build_episodes_list()
        self._last_episode_for_shard = self._compute_shard_eviction_points()
        
        # Announce extraction plan up front by probing one source video.
        if self._episodes_to_import:
            if self._check_needs_reencode():
                print(
                    "\nSource videos require re-encoding to H.264 for "
                    "browser playback (source codec is not H.264/yuv420p)."
                )
            else:
                print(
                    "\nSource videos are already H.264/yuv420p; "
                    "using fast stream copy."
                )
        
        # Upper bound on sample count (one per camera per episode); the
        # actual count may be lower if any source videos are missing.
        approx_samples = len(self._episodes_to_import) * len(self.camera_views)
        print(
            f"Setup complete: ~{approx_samples} samples from "
            f"{len(self._episodes_to_import)} episodes "
            f"(extraction will run lazily during import)"
        )
    
    def _compute_shard_eviction_points(self) -> Dict[Tuple[int, int], int]:
        """Return {(chunk_idx, file_idx): last_episode_index_using_shard}.
        
        Used by ``_iter_sample_groups`` to drop cached PyArrow tables
        as soon as they're no longer needed, keeping memory bounded.
        """
        return {
            self._episode_shard_key(ep): ep["episode_index"]
            for ep in self._episodes_to_import
        }
    
    # Required subdirectories for a valid v3.0 dataset layout
    _REQUIRED_SUBDIRS = (
        ("data", "Parquet data files"),
        ("videos", "Video files"),
        ("meta/episodes", "Episode metadata"),
    )
    
    def _validate_v3_structure(self) -> Dict:
        """Validate the on-disk dataset layout and return the parsed ``info.json``.
        
        Returning the parsed dict here lets ``_load_metadata`` reuse it
        rather than re-reading the file.
        """
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
                f"Please convert your dataset using LeRobot's conversion tools: "
                f"https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py"
            )
        
        for dir_path, description in self._REQUIRED_SUBDIRS:
            full_path = self.dataset_dir / dir_path
            if not full_path.exists():
                raise ValueError(
                    f"Missing required directory: {full_path} ({description})"
                )
        
        print(f"Validated v3.0 dataset at {self.dataset_dir}")
        return info
    
    def _load_metadata(self, info: Dict):
        """Load all metadata from the dataset and pre-compute derived caches.
        
        Args:
            info: Pre-parsed ``info.json`` contents (from
                ``_validate_v3_structure``) so we don't read the file twice.
        """
        meta_dir = self.dataset_dir / "meta"
        
        self._dataset_info = info
        self._features = self._dataset_info.get("features", {})
        self._fps = int(self._dataset_info.get("fps", 30))
        self._video_path_template = self._dataset_info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
        self._data_path_template = self._dataset_info.get(
            "data_path",
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        )
        
        # stats.json holds normalization statistics for ML training (optional)
        stats_path = meta_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self._stats = json.load(f)
        
        # Always discover the full slice-name -> feature-key map up front;
        # then resolve (or filter to) whatever the user asked for.
        self._build_video_feature_map()
        self.camera_views = self._resolve_camera_views(self.camera_views)
        if not self.camera_views:
            raise ValueError(
                f"No camera views available in {self.dataset_dir}. "
                f"Found no 'video' features in info.json and no "
                f"subdirectories in videos/."
            )
        if self.default_slice is None:
            self.default_slice = self.camera_views[0]
        
        self._build_per_camera_caches()
        self._build_frame_field_schema()
        
        # Episode metadata (chunked parquet under meta/episodes/)
        episodes_dir = meta_dir / "episodes"
        episode_files = sorted(episodes_dir.glob("**/*.parquet"))
        if not episode_files:
            raise ValueError(f"No episode metadata found in {episodes_dir}")
        
        episodes_table = pa.concat_tables(
            [pq.read_table(f) for f in episode_files]
        )
        self._episodes = episodes_table.to_pylist()
        
        # Task vocabulary — v3.0 has shipped in three layouts over time:
        #   tasks.parquet   single file (most common in practice)
        #   tasks.jsonl     single file (per official docs)
        #   tasks/*.parquet chunked directory (for very large vocabularies)
        tasks_parquet = meta_dir / "tasks.parquet"
        tasks_jsonl = meta_dir / "tasks.jsonl"
        tasks_dir = meta_dir / "tasks"
        
        if tasks_parquet.exists():
            tbl = pq.read_table(tasks_parquet)
            self._load_task_mapping(tbl.to_pylist(), tbl.column_names)
        elif tasks_jsonl.exists():
            with open(tasks_jsonl) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._task_mapping[row.get("task", "")] = int(
                            row.get("task_index", 0)
                        )
        elif tasks_dir.exists():
            task_files = sorted(tasks_dir.glob("**/*.parquet"))
            if task_files:
                tbl = pa.concat_tables([pq.read_table(f) for f in task_files])
                self._load_task_mapping(tbl.to_pylist(), tbl.column_names)
        
        # Reverse mapping for index -> string lookups (used when episode
        # metadata lacks a 'tasks' field and we have to decode per-frame
        # task_index values back to their natural-language task string).
        self._tasks_by_index = {v: k for k, v in self._task_mapping.items()}
        
        # Scan data shards to build the trusted episode -> shard map.
        self._build_episode_location_map()
        
        print(
            f"Loaded metadata: {len(self._episodes)} episodes, "
            f"{len(self.camera_views)} cameras, "
            f"{len(self._task_mapping)} tasks, FPS={self._fps}"
        )
    
    def _build_episode_location_map(self):
        """Scan every data parquet shard once to map episodes to their real location.
        
        Reads only the ``episode_index`` column from each shard (cheap), then
        walks the column to identify contiguous runs and record each
        episode's within-shard slice offsets. The resulting map is the only
        reliable way to read frame data when episode metadata's
        ``data/file_index`` or ``dataset_from_index`` fields are stale or
        global-rather-than-shard-local (as in ``lerobot/libero``).
        """
        data_dir = self.dataset_dir / "data"
        shard_files = sorted(data_dir.glob("**/*.parquet"))
        if not shard_files:
            return
        
        for path in shard_files:
            shard_key = self._parse_shard_key_from_path(path)
            if shard_key is None:
                continue
            
            table = pq.read_table(path, columns=["episode_index"])
            ep_col = table.column("episode_index").to_pylist()
            if not ep_col:
                continue
            
            # Walk the column once, recording contiguous (episode, start, end)
            # runs. Within a shard, each episode's rows are contiguous.
            current_ep = ep_col[0]
            run_start = 0
            for i in range(1, len(ep_col)):
                if ep_col[i] != current_ep:
                    self._episode_location[int(current_ep)] = (
                        shard_key, run_start, i
                    )
                    current_ep = ep_col[i]
                    run_start = i
            self._episode_location[int(current_ep)] = (
                shard_key, run_start, len(ep_col)
            )
    
    @staticmethod
    def _parse_shard_key_from_path(path: Path) -> Optional[Tuple[int, int]]:
        """Extract ``(chunk_index, file_index)`` from a ``chunk-NNN/file-MMM.parquet`` path."""
        try:
            chunk_idx = int(path.parent.name.replace("chunk-", ""))
            file_idx = int(path.stem.replace("file-", ""))
        except ValueError:
            return None
        return chunk_idx, file_idx
    
    def _build_per_camera_caches(self):
        """Pre-compute per-camera lookup keys and frame dimensions.
        
        Run once after ``camera_views`` and ``_video_feature_map`` are known
        so the hot path in ``_create_episode_samples`` does only dict lookups.
        """
        for camera in self.camera_views:
            # _resolve_camera_views guarantees every entry is in the map
            video_key = self._video_feature_map[camera]
            self._camera_episode_keys[camera] = (
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
                f"videos/{video_key}/from_timestamp",
                f"videos/{video_key}/to_timestamp",
            )
            shape = self._features.get(video_key, {}).get("shape", [])
            height = shape[0] if len(shape) > 0 else None
            width = shape[1] if len(shape) > 1 else None
            self._camera_dimensions[camera] = (height, width)
    
    def _episode_shard_key(self, episode: Dict) -> Tuple[int, int]:
        """Return the (chunk_index, file_index) tuple for an episode's data shard.
        
        Prefers the trusted ``_episode_location`` map (built by scanning
        shard contents at setup). Falls back to the episode metadata's
        ``data/chunk_index`` + ``data/file_index`` only if the scan didn't
        cover this episode (which shouldn't happen for a well-formed
        dataset, but the fallback keeps the method total).
        """
        location = self._episode_location.get(int(episode["episode_index"]))
        if location is not None:
            return location[0]
        chunk = episode.get("data/chunk_index", episode.get("data_chunk_index", 0))
        file_ = episode.get("data/file_index", episode.get("data_file_index", 0))
        return int(chunk), int(file_)
    
    def _episode_row_range(self, episode: Dict) -> Tuple[int, int]:
        """Return ``(from_row, to_row)`` within-shard slice offsets for ``episode``.
        
        Prefers the trusted ``_episode_location`` map; falls back to the
        episode metadata's ``dataset_from_index`` / ``dataset_to_index``
        (only correct when there's a single shard or the values happen to
        align with shard boundaries).
        """
        location = self._episode_location.get(int(episode["episode_index"]))
        if location is not None:
            _, from_row, to_row = location
            return from_row, to_row
        from_row = int(episode.get("dataset_from_index", 0))
        to_row = int(episode.get(
            "dataset_to_index", from_row + episode.get("length", 0)
        ))
        return from_row, to_row
    
    def _load_task_mapping(self, rows: List[Dict], column_names: List[str]):
        """Build the ``task_string -> task_index`` mapping from task rows.
        
        Two parquet layouts are seen in the wild:
        
        - Standard: columns ``"task"`` and ``"task_index"``.
        - Pandas-style: ``"task_index"`` plus a string column holding the task
          text (whatever its name is — pandas may have written the index out
          under any label).
        """
        if "task" in column_names and "task_index" in column_names:
            for row in rows:
                self._task_mapping[str(row["task"])] = int(row["task_index"])
            return
        
        if "task_index" not in column_names:
            return
        
        task_col = next((c for c in column_names if c != "task_index"), None)
        if task_col is None:
            return
        for row in rows:
            self._task_mapping[str(row[task_col])] = int(row["task_index"])
    
    def _build_video_feature_map(self):
        """Populate ``_video_feature_map`` (slice name -> full feature key).
        
        Always runs (regardless of whether the user supplied ``camera_views``)
        so resolution is uniform downstream. Slice names are derived from the
        video feature keys by stripping the longest common dot-segment prefix
        and joining the remainder with underscores (dots are reserved by
        MongoDB for nested field access).
        
        Examples:
        
        - ``["observation.images.top", "observation.images.wrist"]``
          -> ``{"top": ..., "wrist": ...}``
        - ``["observation.images.wrist.top", "observation.images.top.front"]``
          -> ``{"wrist_top": ..., "top_front": ...}``
        - ``["observation.images.rgb.head", "observation.images.depth.head"]``
          -> ``{"rgb_head": ..., "depth_head": ...}``
        - ``["observation.images.front_view"]`` -> ``{"front_view": ...}``
        """
        video_keys = [
            key for key, feat in self._features.items()
            if feat.get("dtype") == "video"
        ]
        
        # Fallback: when info.json lacks video features, scan the videos dir.
        # In this fallback world the subdir name is also the "feature key".
        if not video_keys:
            videos_dir = self.dataset_dir / "videos"
            if videos_dir.exists():
                video_keys = [
                    d.name for d in videos_dir.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
        
        if not video_keys:
            return
        
        if len(video_keys) == 1:
            key = video_keys[0]
            self._video_feature_map[key.split(".")[-1]] = key
            return
        
        # Strip the longest common leading dot-segment prefix.
        parts_list = [key.split(".") for key in video_keys]
        prefix_len = 0
        for segments in zip(*parts_list):
            if len(set(segments)) != 1:
                break
            prefix_len += 1
        
        for key in video_keys:
            self._video_feature_map["_".join(key.split(".")[prefix_len:])] = key
    
    def _resolve_camera_views(
        self, requested: Optional[List[str]]
    ) -> List[str]:
        """Return the final list of slice names to import.
        
        If ``requested`` is None, return all auto-detected slices (sorted).
        Otherwise resolve each requested entry against:
        
        1. a known slice name (the keys of ``_video_feature_map``),
        2. a full feature key (the values), or
        3. a videos-subdir-style name (the legacy fallback case).
        
        Raises ``ValueError`` with a useful "did you mean ..." message if any
        entry doesn't resolve, so users don't silently get a 0-sample import.
        """
        if requested is None:
            return sorted(self._video_feature_map)
        
        # Reverse lookup: feature key -> slice name
        key_to_slice = {v: k for k, v in self._video_feature_map.items()}
        
        resolved: List[str] = []
        unknown: List[str] = []
        for name in requested:
            if name in self._video_feature_map:
                resolved.append(name)
            elif name in key_to_slice:
                resolved.append(key_to_slice[name])
            else:
                unknown.append(name)
        
        if unknown:
            available = sorted(self._video_feature_map)
            raise ValueError(
                f"Unknown camera_views entries: {unknown}. "
                f"Available slice names: {available}. "
                f"You may also pass full feature keys: "
                f"{sorted(self._video_feature_map.values())}"
            )
        return resolved
    
    def _should_include_field(self, lerobot_name: str) -> bool:
        """Return True if ``lerobot_name`` passes the include/exclude filters.
        
        Patterns are ``fnmatch`` globs evaluated against the original
        LeRobot dot-notation name (e.g. ``"observation.state"``).
        """
        if self.include_fields is not None and not any(
            fnmatch.fnmatch(lerobot_name, p) for p in self.include_fields
        ):
            return False
        if self.exclude_fields is not None and any(
            fnmatch.fnmatch(lerobot_name, p) for p in self.exclude_fields
        ):
            return False
        return True
    
    # Fields stored on the sample (or reconstructible from frame numbering)
    # rather than as per-frame FiftyOne fields
    _SKIP_FRAME_FIELDS = frozenset(
        {"episode_index", "frame_index", "index", "task_index"}
    )
    
    def _build_frame_field_schema(self):
        """
        Build the frame field schema from ``info.json`` features.
        
        Each non-video, non-skipped feature becomes one FiftyOne frame field.
        Type derivation (info.json dtype + shape -> FiftyOne field type):
        
            float32/64 shape=[1] -> FloatField    (Python float)
            float32/64 shape=[N] -> ListField     (Python list[float])
            int64      shape=[1] -> IntField      (Python int)
            int64      shape=[N] -> ListField     (Python list[int])
            bool       shape=[1] -> BooleanField  (Python bool)
            bool       shape=[N] -> ListField     (Python list[bool])
        """
        scalar_count = 0
        for lerobot_name, feat_def in self._features.items():
            dtype = feat_def.get("dtype", "")
            if dtype == "video" or lerobot_name in self._SKIP_FRAME_FIELDS:
                continue
            if not self._should_include_field(lerobot_name):
                continue
            
            shape = feat_def.get("shape", [1])
            # FiftyOne field names cannot contain dots (MongoDB nested access)
            fo_name = lerobot_name.replace(".", "_")
            
            self._frame_fields[fo_name] = {
                "lerobot_name": lerobot_name,
                "dtype": dtype,
                "shape": shape,
            }
            if math.prod(shape) == 1:
                scalar_count += 1
            
            names = feat_def.get("names")
            if names:
                self._field_names_meta[fo_name] = names
            
            description = feat_def.get("description")
            desc_parts = []
            if description:
                desc_parts.append(description)
            if names:
                desc_parts.append("[" + ", ".join(names) + "]")
            if desc_parts:
                self._field_descriptions[fo_name] = " : ".join(desc_parts)
        
        # Parquet column projection: only read columns we'll actually use
        self._columns_to_read = [
            fi["lerobot_name"] for fi in self._frame_fields.values()
        ]
        
        total = len(self._frame_fields)
        print(
            f"Frame field schema: {total} fields "
            f"({scalar_count} scalar, {total - scalar_count} array)"
        )
    
    def _build_episodes_list(self):
        """Apply user-supplied filters and produce the episode iteration order."""
        episodes = list(self._episodes)
        
        if self.episode_ids is not None:
            wanted = set(self.episode_ids)
            episodes = [e for e in episodes if e["episode_index"] in wanted]
        
        if self.task_ids is not None:
            wanted = set(self.task_ids)
            episodes = [e for e in episodes if e.get("task_index") in wanted]
        
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(episodes)
        
        if self.max_samples is not None:
            episodes = episodes[: self.max_samples]
        
        self._episodes_to_import = episodes
    
    def _check_needs_reencode(self) -> bool:
        """Probe the first episode's first camera to decide whether the
        source codec requires re-encoding.
        
        Returns:
            True if at least one re-encode is expected, False if stream copy
            will work. Returns False for empty datasets or missing sources
            (the per-clip path will skip them with a warning anyway).
        """
        if not self._episodes_to_import or not self.camera_views:
            return False
        
        episode = self._episodes_to_import[0]
        camera = self.camera_views[0]
        chunk_key, file_key, _, _ = self._camera_episode_keys[camera]
        if chunk_key not in episode:
            return False
        
        source_video = self._resolve_video_path(
            camera, int(episode[chunk_key]), int(episode[file_key])
        )
        if not source_video.exists():
            return False
        
        return not self._video_splitter.is_browser_compatible(source_video)
    
    def _iter_sample_groups(self) -> Iterator[List[Dict]]:
        """Lazily yield one episode's worth of sample dicts at a time.
        
        Performs clip extraction and parquet reads on demand. Evicts both
        parquet shard caches (frame data + task_index fallback) once the
        last episode using a shard has been emitted, keeping memory bounded
        for large datasets.
        """
        for episode in self._episodes_to_import:
            samples = self._create_episode_samples(episode)
            
            shard_key = self._episode_shard_key(episode)
            if self._last_episode_for_shard.get(shard_key) == episode["episode_index"]:
                self._data_table_cache.pop(shard_key, None)
                self._task_index_cache.pop(shard_key, None)
            
            if samples:
                yield samples
    
    def _resolve_video_path(self, camera: str, chunk_idx: int, file_idx: int) -> Path:
        """Resolve the on-disk path to a sharded source MP4 for ``camera``."""
        rel_path = self._video_path_template.format(
            video_key=self._video_feature_map[camera],
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        return self.dataset_dir / rel_path
    
    def _resolve_data_path(self, chunk_idx: int, file_idx: int) -> Path:
        """Resolve the on-disk path to a sharded parquet data file."""
        rel_path = self._data_path_template.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        return self.dataset_dir / rel_path
    
    def _load_episode_frame_data(self, episode: Dict):
        """Load and slice the per-frame parquet rows for one episode.
        
        Uses column pruning (only reads ``_columns_to_read``) and caches the
        full shard between episodes — multiple episodes typically share a
        shard, so this turns N parquet reads into M shard reads (M << N).
        Slice offsets come from the trusted ``_episode_location`` map (see
        ``_build_episode_location_map``) rather than the often-stale
        ``dataset_from_index`` / ``dataset_to_index`` fields in episode
        metadata.
        
        Returns:
            A PyArrow Table slice for this episode, or ``None`` if the shard
            is missing on disk.
        """
        shard_key = self._episode_shard_key(episode)
        from_row, to_row = self._episode_row_range(episode)
        
        table = self._data_table_cache.get(shard_key)
        if table is None:
            parquet_path = self._resolve_data_path(*shard_key)
            if not parquet_path.exists():
                print(f"Warning: Parquet file not found: {parquet_path}")
                return None
            
            available_cols = set(pq.read_schema(parquet_path).names)
            columns = [c for c in self._columns_to_read if c in available_cols]
            table = pq.read_table(parquet_path, columns=columns)
            self._data_table_cache[shard_key] = table
        
        return table.slice(from_row, to_row - from_row)
    
    def _resolve_episode_task(
        self, episode: Dict
    ) -> Tuple[Optional[str], Optional[int]]:
        """Return ``(task_string, task_index)`` for one episode.
        
        Tries layouts in priority order:
        
        1. **Episode-level ``tasks`` list** (canonical v3.0): the first entry
           is the natural-language task; we look up its index in the
           ``meta/tasks*`` vocabulary.
        2. **Per-frame ``task_index`` column** (as in ``lerobot/libero``):
           when the episode metadata has no ``tasks`` field, we peek at the
           first frame's ``task_index`` and reverse-look-up the string via
           the tasks vocabulary.
        
        Returns ``(None, None)`` if neither layout yields task info.
        """
        # Layout 1: episode["tasks"] = [task_str, ...]
        tasks_list = episode.get("tasks") or []
        if tasks_list:
            task_string = tasks_list[0]
            return task_string, self._task_mapping.get(task_string)
        
        # Layout 2: derive from per-frame task_index column
        task_idx = self._episode_first_task_index(episode)
        if task_idx is None:
            return None, None
        return self._tasks_by_index.get(task_idx), task_idx
    
    def _episode_first_task_index(self, episode: Dict) -> Optional[int]:
        """Read the first frame's ``task_index`` for ``episode`` from parquet.
        
        Used as a fallback when episode metadata lacks the canonical
        episode-level ``tasks`` list (some v3.0 datasets, e.g. ``lerobot/libero``,
        store task assignment only as a per-frame column in the data parquet).
        
        Uses the broader ``_data_table_cache`` if it already contains the
        ``task_index`` column. Otherwise does a tiny one-column projection
        read cached separately so per-episode lookups stay O(1) after the
        first hit on a shard. Within-shard offsets come from the trusted
        ``_episode_location`` map.
        
        Returns ``None`` if the parquet shard is missing or has no
        ``task_index`` column.
        """
        shard_key = self._episode_shard_key(episode)
        from_row, _ = self._episode_row_range(episode)
        
        # Path 1: piggyback on the frame-data cache if task_index is in there
        frame_table = self._data_table_cache.get(shard_key)
        if frame_table is not None and "task_index" in frame_table.column_names:
            value = frame_table.column("task_index")[from_row].as_py()
            return None if value is None else int(value)
        
        # Path 2: cheap dedicated projection read, cached per shard
        task_table = self._task_index_cache.get(shard_key)
        if task_table is None:
            parquet_path = self._resolve_data_path(*shard_key)
            if not parquet_path.exists():
                return None
            available = set(pq.read_schema(parquet_path).names)
            if "task_index" not in available:
                return None
            task_table = pq.read_table(parquet_path, columns=["task_index"])
            self._task_index_cache[shard_key] = task_table
        
        if from_row >= task_table.num_rows:
            return None
        value = task_table.column("task_index")[from_row].as_py()
        return None if value is None else int(value)
    
    def _extract_frame_columns(self, frame_data) -> Dict[str, List]:
        """Pull per-frame Python value lists from a PyArrow table once.
        
        Robot states / actions are identical across cameras of the same
        episode, so we extract each column once per episode and reuse the
        resulting lists for every camera sample in the group.
        
        ``to_pylist()`` returns native Python types directly
        (float32/64 -> float, int64 -> int, bool -> bool, list<T> -> list)
        with no numpy intermediary.
        """
        if frame_data is None:
            return {}
        column_names = set(frame_data.column_names)
        return {
            fo_name: frame_data.column(fi["lerobot_name"]).to_pylist()
            for fo_name, fi in self._frame_fields.items()
            if fi["lerobot_name"] in column_names
        }
    
    def _create_episode_samples(self, episode: Dict) -> List[Dict]:
        """Build sample dicts for every camera view of one episode."""
        episode_idx = episode["episode_index"]
        group_id = focu.ObjectId()
        episode_length = int(episode.get("length", 0))
        
        task_string, task_index = self._resolve_episode_task(episode)
        
        # Extract per-frame columns ONCE per episode; all cameras share them.
        frame_columns: Dict[str, List] = {}
        if self.include_frame_data:
            frame_columns = self._extract_frame_columns(
                self._load_episode_frame_data(episode)
            )
        
        common = {
            "group_id": group_id,
            "episode_index": episode_idx,
            "task": task_string,
            "task_index": task_index,
            "dataset_from_index": episode.get("dataset_from_index"),
            "dataset_to_index": episode.get("dataset_to_index"),
            "frame_columns": frame_columns,
        }
        
        samples = []
        for camera in self.camera_views:
            chunk_key, file_key, from_key, to_key = self._camera_episode_keys[camera]
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
            except (RuntimeError, ValueError, FileNotFoundError) as e:
                print(
                    f"Warning: Failed to extract episode {episode_idx} "
                    f"camera {camera}: {e}"
                )
                continue
            
            frame_height, frame_width = self._camera_dimensions[camera]
            video_metadata = fom.VideoMetadata(
                frame_width=frame_width,
                frame_height=frame_height,
                frame_rate=float(self._fps),
                total_frame_count=episode_length,
                duration=to_ts - from_ts,
                mime_type="video/mp4",
                encoding_str="avc1",  # output is always H.264, copy or re-encoded
            )
            
            samples.append({
                **common,
                "filepath": str(clip_path),
                "group_name": camera,
                "camera_view": camera,
                "metadata": video_metadata,
            })
        
        return samples
    
    def _apply_frame_columns(self, sample: fo.Sample, frame_columns: Dict[str, List]):
        """Assign pre-extracted per-frame value lists onto a sample's frames.
        
        FiftyOne frames are 1-indexed; ``None`` values are left unset so the
        field stays default rather than being explicitly nulled.
        """
        for fo_name, values in frame_columns.items():
            for i, val in enumerate(values, start=1):
                if val is not None:
                    sample.frames[i][fo_name] = val
    
    def __len__(self) -> int:
        """Return number of groups (episodes) to import."""
        if self._episodes_to_import is None:
            return 0
        return len(self._episodes_to_import)
    
    def __iter__(self):
        """Begin (or restart) lazy iteration over episode groups.
        
        Builds a fresh generator each time so that repeated iteration is
        well-defined. Extraction and parquet reads happen inside the
        generator on demand.
        """
        self._samples_iter = self._iter_sample_groups()
        return self
    
    # Optional sample-level fields copied verbatim from sample_dict
    _OPTIONAL_SAMPLE_FIELDS = ("task", "task_index")
    _INT_SAMPLE_FIELDS = ("dataset_from_index", "dataset_to_index")
    
    def __next__(self) -> Dict[str, fo.Sample]:
        """Return the next group of samples (one per camera view of one episode).
        
        Returns:
            Dict mapping slice (camera) name to ``fo.Sample`` instances.
        """
        sample_dicts = next(self._samples_iter)
        group_samples: Dict[str, fo.Sample] = {}
        
        for sd in sample_dicts:
            sample = fo.Sample(filepath=sd["filepath"])
            sample.metadata = sd["metadata"]
            sample["episode_index"] = sd["episode_index"]
            sample["camera_view"] = sd["camera_view"]
            
            for key in self._OPTIONAL_SAMPLE_FIELDS:
                value = sd.get(key)
                if value is not None:
                    sample[key] = value
            for key in self._INT_SAMPLE_FIELDS:
                value = sd.get(key)
                if value is not None:
                    sample[key] = int(value)
            
            sample[self._group_field] = fog.Group(
                id=sd["group_id"], name=sd["group_name"]
            )
            
            if sd["frame_columns"]:
                self._apply_frame_columns(sample, sd["frame_columns"])
            
            group_samples[sd["group_name"]] = sample
        
        return group_samples
    
    def get_group_media_types(self) -> Dict[str, str]:
        """Return media types for each group slice."""
        return {camera: "video" for camera in self.camera_views}
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return the full LeRobot metadata bundle for ML training and export.
        
        This is attached to ``dataset.info`` by FiftyOne's ``add_importer``.
        Includes the original ``features`` definitions, normalization stats,
        task vocabulary, and the slice-name -> feature-key mapping needed to
        round-trip back to the LeRobot layout on export.
        """
        if self._dataset_info is None:
            return {}
        
        return {
            "type": "LeRobot v3.0 Dataset",
            "format": "grouped_video",
            "codebase_version": self._dataset_info.get("codebase_version", "v3.0"),
            "robot_type": self._dataset_info.get("robot_type"),
            "episode_count": len(self._episodes_to_import or ()),
            "total_episodes": self._dataset_info.get("total_episodes", 0),
            "total_frames": self._dataset_info.get("total_frames", 0),
            "camera_views": self.camera_views,
            "default_slice": self.default_slice,
            "group_field": self._group_field,
            "fps": self._fps,
            "features": self._features,
            "stats": self._stats,
            "tasks": self._task_mapping,
            "video_feature_map": dict(self._video_feature_map),
        }
    
    # Static descriptions for sample-level fields written by this importer
    _SAMPLE_FIELD_DESCRIPTIONS: Dict[str, str] = {
        "episode_index": "Episode number within the dataset",
        "camera_view": "Camera view / group slice name",
        "task": "Task description string",
        "task_index": "Task index in the task vocabulary",
        "dataset_from_index": "Start row index in the source LeRobot parquet file",
        "dataset_to_index": "End row index in the source LeRobot parquet file",
    }
    
    def apply_field_descriptions(self, dataset: fo.Dataset):
        """Instance-method shim that calls :func:`apply_lerobot_field_descriptions`."""
        apply_lerobot_field_descriptions(dataset)
    
    def close(self, *args):
        """Clean up resources."""
        self._data_table_cache.clear()
        self._task_index_cache.clear()
        self._samples_iter = None


def apply_lerobot_field_descriptions(dataset: fo.Dataset) -> None:
    """Attach LeRobot-derived descriptions and metadata to fields on ``dataset``.
    
    Reads ``dataset.info["features"]`` (populated automatically by
    :class:`LeRobotDatasetImporter` via ``get_dataset_info``) and applies:
    
    - ``field.description`` -- ``"<info.json description> : [n1, n2, ...]"``
    - ``field.info`` -- ``{"lerobot_name", "dtype", "shape", "names"?}`` so
      the original LeRobot identity is preserved for round-trip export
    
    Static descriptions are also applied to the sample-level fields this
    importer writes (``episode_index``, ``task``, etc.).
    
    Safe to call multiple times. Save failures on individual fields are
    warned about but not raised, so partial schemas don't abort the
    annotation pass.
    
    This function is normally invoked automatically by
    :func:`import_lerobot_dataset`. Users of ``fo.Dataset.from_dir()``
    should call it explicitly after import to attach field metadata::
    
        dataset = fo.Dataset.from_dir(
            dataset_dir="...", dataset_type=LeRobotDataset, name="..."
        )
        apply_lerobot_field_descriptions(dataset)
    """
    info = dataset.info or {}
    features = info.get("features", {})
    if not features:
        warnings.warn(
            "dataset.info has no 'features' entry; was this dataset imported "
            "with LeRobotDataset? Skipping field descriptions."
        )
        return
    
    # Frame fields (anything in features that isn't a video and isn't a
    # sample-level skip field). Mirrors _build_frame_field_schema.
    for lerobot_name, feat_def in features.items():
        dtype = feat_def.get("dtype", "")
        if (
            dtype == "video"
            or lerobot_name in LeRobotDatasetImporter._SKIP_FRAME_FIELDS
        ):
            continue
        
        fo_name = lerobot_name.replace(".", "_")
        field = dataset.get_field(f"frames.{fo_name}")
        if field is None:
            # The user may have filtered this field out via include/exclude
            continue
        
        names = feat_def.get("names")
        description = feat_def.get("description")
        desc_parts = []
        if description:
            desc_parts.append(description)
        if names:
            desc_parts.append("[" + ", ".join(names) + "]")
        if desc_parts:
            field.description = " : ".join(desc_parts)
        
        field.info = {
            "lerobot_name": lerobot_name,
            "dtype": dtype,
            "shape": feat_def.get("shape", []),
        }
        if names:
            field.info["names"] = names
        
        try:
            field.save()
        except Exception as e:
            warnings.warn(f"Failed to save metadata for frames.{fo_name}: {e}")
    
    # Sample-level field descriptions
    for field_name, description in (
        LeRobotDatasetImporter._SAMPLE_FIELD_DESCRIPTIONS.items()
    ):
        field = dataset.get_field(field_name)
        if field is None:
            continue
        field.description = description
        try:
            field.save()
        except Exception as e:
            warnings.warn(f"Failed to save description for {field_name}: {e}")


class LeRobotDataset(fot.Dataset):
    """Dataset type for LeRobot v3.0 robotics datasets.
    
    Use with ``fo.Dataset.from_dir()``::
    
        dataset = fo.Dataset.from_dir(
            dataset_dir="/path/to/dataset",
            dataset_type=LeRobotDataset,
            camera_views=["cam_high", "cam_low"],
            name="my_dataset",
        )
        # `from_dir` doesn't propagate per-field descriptions automatically;
        # call this once afterwards to attach them:
        apply_lerobot_field_descriptions(dataset)
    
    Or use :func:`import_lerobot_dataset`, which calls
    :func:`apply_lerobot_field_descriptions` for you.
    """
    
    def get_dataset_importer_cls(self):
        return LeRobotDatasetImporter


# Convenience function for direct import
def import_lerobot_dataset(
    dataset_dir: Union[str, Path],
    name: Optional[str] = None,
    camera_views: Optional[List[str]] = None,
    episode_ids: Optional[List[int]] = None,
    task_ids: Optional[List[int]] = None,
    include_frame_data: bool = True,
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
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
        include_fields: Glob patterns for LeRobot field names to include
            (e.g., ["observation.*", "action.*", "timestamp"]).
            None means include all non-video fields.
        exclude_fields: Glob patterns for LeRobot field names to exclude
            (e.g., ["*.is_fresh"]). Applied after include_fields.
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
        if not overwrite:
            raise ValueError(
                f"Dataset '{name}' already exists. "
                f"Use overwrite=True to replace it."
            )
        fo.delete_dataset(name)
    
    # Drive the importer directly (rather than via from_dir) so we get a
    # handle on it for the post-import apply_field_descriptions step.
    importer = LeRobotDatasetImporter(
        dataset_dir=str(dataset_dir),
        camera_views=camera_views,
        episode_ids=episode_ids,
        task_ids=task_ids,
        include_frame_data=include_frame_data,
        include_fields=include_fields,
        exclude_fields=exclude_fields,
        max_samples=max_samples,
        **kwargs,
    )
    
    dataset = fo.Dataset(name)
    dataset.add_importer(importer, dynamic=True)
    apply_lerobot_field_descriptions(dataset)
    
    return dataset


# Register the dataset type with FiftyOne so it's discoverable as fot.LeRobotDataset
fot.LeRobotDataset = LeRobotDataset