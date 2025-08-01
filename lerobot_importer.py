"""
FiftyOne importer for extracted LeRobot dataset format.

This importer is specifically designed for LeRobot datasets that have been
extracted into individual PNG images and JSON metadata files, with a separate
meta/ directory containing episode and task information.
"""

import os
import json
import jsonlines
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import uuid
import random

import fiftyone as fo
import fiftyone.core.dataset as foud
from fiftyone.utils.data.importers import LabeledImageDatasetImporter, ImportPathsMixin
import fiftyone.core.labels as fol
import fiftyone.core.metadata as fom
import fiftyone.core.utils as focu
import fiftyone.core.fields as fof


class LeRobotDatasetImporter(
    LabeledImageDatasetImporter, 
    ImportPathsMixin):
    """
    Importer for extracted LeRobot format robotics datasets.
    
    This importer is opinionated about the dataset structure and only works with
    datasets that have been extracted into individual PNG images and JSON files.
    
    Expected structure:
    ```
    dataset_root/
    ├── extracted_data/
    │   ├── episode_000000/
    │   │   ├── episode_000000_000000_cam_low.png
    │   │   ├── episode_000000_000000_cam_high.png
    │   │   ├── episode_000000_000000_cam_right_wrist.png
    │   │   ├── episode_000000_000000_cam_left_wrist.png
    │   │   ├── episode_000000_000000.json
    │   │   └── ... (more frames)
    │   └── episode_000001/
    └── meta/
        ├── info.json
        ├── episodes.jsonl
        ├── tasks.jsonl
        └── stats.json
    ```
    
    Args:
        dataset_dir (None): root directory containing extracted_data/ and meta/
        data_path (None): path to extracted_data directory (if different from dataset_dir/extracted_data)
        labels_path (None): path to meta directory (if different from dataset_dir/meta)
        camera_views (required): list of camera view names (e.g., ["low", "high", "right_wrist"])
        episode_ids (None): list of specific episode IDs to import. If None, imports all episodes
        task_ids (None): list of specific task IDs to import. If None, imports all tasks
        include_metadata (True): whether to load trajectory metadata from JSON files
        max_samples (None): maximum number of frame groups to import
        shuffle (False): whether to shuffle the import order
        seed (None): random seed for shuffling
        default_slice (None): default camera view slice for grouped dataset
        group_field ("group"): name of the group field in the dataset
    """
    
    def __init__(
        self,
        dataset_dir=None,
        data_path=None,
        labels_path=None,
        camera_views=None,
        episode_ids=None,
        task_ids=None,
        include_metadata=True,
        max_samples=None,
        shuffle=False,
        seed=None,
        default_slice=None,
        group_field="group",
        **kwargs
    ):
        # Validate required parameters
        if dataset_dir is None and (data_path is None or labels_path is None):
            raise ValueError(
                "Either 'dataset_dir' must be provided, or both 'data_path' and 'labels_path' must be provided"
            )
        
        if camera_views is None or len(camera_views) == 0:
            raise ValueError(
                "camera_views is required and must be a non-empty list of camera view names "
                "(e.g., ['low', 'high', 'right_wrist', 'left_wrist'])"
            )
        
        # Parse paths
        if dataset_dir is not None:
            dataset_dir = os.path.abspath(dataset_dir)
            data_path = data_path or os.path.join(dataset_dir, "extracted_data")
            labels_path = labels_path or os.path.join(dataset_dir, "meta")
        
        # Call parent constructor
        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )
        
        # Store parameters
        self.data_path = data_path
        self.labels_path = labels_path
        self.camera_views = camera_views
        self.episode_ids = episode_ids
        self.task_ids = task_ids
        self.include_metadata = include_metadata
        self.default_slice = default_slice or camera_views[0]
        self.group_field = group_field
        
        # Internal state
        self._dataset_info = None
        self._episodes_info = None
        self._tasks_info = None
        self._episodes_stats = None
        self._extracted_frames = None
        self._frame_groups = None
        self._current_idx = 0

    @property
    def has_dataset_info(self):
        """Whether this importer produces a dataset info dict."""
        return True

    @property
    def has_image_metadata(self):
        """Whether this importer produces image metadata."""
        return True

    @property 
    def label_cls(self):
        """The label class(es) produced by this importer."""
        return None  # This is a grouped dataset with trajectory metadata

    def setup(self):
        """Setup method called before iteration begins."""
        # Validate paths
        data_path = Path(self.data_path)
        labels_path = Path(self.labels_path)
        
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        if not labels_path.exists():
            raise ValueError(f"Labels path does not exist: {labels_path}")
        
        # Load metadata from meta/ directory
        self._load_metadata(labels_path)
        
        # Collect extracted format files
        self._extracted_frames = self._collect_extracted_files(data_path)
        
        # Apply filtering and generate frame groups
        self._apply_filters_and_generate_groups()
        
        # Initialize iteration state
        self._current_idx = 0

    def _load_metadata(self, labels_path: Path):
        """Load metadata from meta/ directory."""
        # Load info.json
        info_path = labels_path / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                self._dataset_info = json.load(f)
        else:
            print(f"Warning: info.json not found at {info_path}")
            self._dataset_info = {}
        
        # Load episodes.jsonl
        episodes_path = labels_path / "episodes.jsonl"
        if episodes_path.exists():
            self._episodes_info = {}
            with jsonlines.open(episodes_path) as reader:
                for episode_data in reader:
                    episode_idx = episode_data["episode_index"]
                    self._episodes_info[episode_idx] = episode_data
        else:
            print(f"Warning: episodes.jsonl not found at {episodes_path}")
            self._episodes_info = {}
        
        # Load tasks.jsonl
        tasks_path = labels_path / "tasks.jsonl"
        if tasks_path.exists():
            self._tasks_info = {}
            with jsonlines.open(tasks_path) as reader:
                for task_data in reader:
                    task_idx = task_data["task_index"]
                    self._tasks_info[task_idx] = task_data
        else:
            print(f"Warning: tasks.jsonl not found at {tasks_path}")
            self._tasks_info = {}
        
        # Load stats.json (optional)
        stats_path = labels_path / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self._episodes_stats = json.load(f)
        
        print(f"Loaded metadata: {len(self._episodes_info)} episodes, {len(self._tasks_info)} tasks")

    def _parse_filename(self, filename: str) -> Optional[Tuple[Optional[str]]]:
        """
        Parse filename to extract camera view for images.
        
        Returns:
            tuple: (camera_view,) where camera_view is None for JSON files
        """
        # Pattern for images: episode_000000_000000_cam_low.png
        img_pattern = r'episode_\d{6}_\d{6}_cam_([a-z_]+)\.png'
        img_match = re.match(img_pattern, filename)
        
        if img_match:
            camera_view = img_match.group(1)
            return (camera_view,)
            
        # Pattern for JSON: episode_000000_000000.json
        json_pattern = r'episode_\d{6}_\d{6}\.json'
        json_match = re.match(json_pattern, filename)
        
        if json_match:
            return (None,)  # JSON file, no camera view
            
        return None

    def _collect_extracted_files(self, data_path: Path) -> Dict[str, Dict]:
        """
        Collect all files organized by episode and frame using JSON metadata.
        
        Returns:
            dict: {
                episode_frame_key: {
                    'episode_index': int,
                    'frame_index': int, 
                    'cameras': {camera_view: filepath},
                    'json': json_filepath,
                    'json_data': dict  # Cache JSON data
                }
            }
        """
        frames = {}
        
        # Look for episode directories
        episode_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('episode_')]
        
        if not episode_dirs:
            raise ValueError(f"No episode directories found in {data_path}")
        
        print(f"Found {len(episode_dirs)} episode directories")
        
        for episode_dir in sorted(episode_dirs):
            print(f"Processing {episode_dir.name}...")
            
            # First pass: collect all JSON files to get episode/frame indices
            json_files = {}
            for filepath in episode_dir.iterdir():
                if filepath.suffix == '.json':
                    try:
                        with open(filepath, 'r') as f:
                            json_data = json.load(f)
                        
                        episode_index = json_data.get('episode_index')
                        frame_index = json_data.get('frame_index')
                        
                        if episode_index is not None and frame_index is not None:
                            # Apply episode filtering
                            if self.episode_ids is not None and episode_index not in self.episode_ids:
                                continue
                                
                            frame_key = f"{episode_index:06d}_{frame_index:06d}"
                            json_files[frame_key] = {
                                'filepath': str(filepath),
                                'data': json_data,
                                'episode_index': episode_index,
                                'frame_index': frame_index
                            }
                    except Exception as e:
                        print(f"Warning: Could not parse JSON {filepath}: {e}")
                        continue
            
            # Second pass: match image files to JSON files based on frame keys
            for filepath in episode_dir.iterdir():
                if filepath.suffix == '.png':
                    parsed = self._parse_filename(filepath.name)
                    if parsed is None:
                        continue
                        
                    camera_view = parsed[0]
                    if camera_view is None or camera_view not in self.camera_views:
                        continue
                    
                    # Try to match this image to a JSON file by checking all frame keys
                    # Look for matching pattern in filename
                    filename_pattern = re.match(r'episode_(\d{6})_(\d{6})_cam_', filepath.name)
                    if filename_pattern:
                        file_episode = int(filename_pattern.group(1))
                        file_frame = int(filename_pattern.group(2))
                        frame_key = f"{file_episode:06d}_{file_frame:06d}"
                        
                        if frame_key in json_files:
                            # Initialize frame data if needed
                            if frame_key not in frames:
                                json_info = json_files[frame_key]
                                frames[frame_key] = {
                                    'episode_index': json_info['episode_index'],
                                    'frame_index': json_info['frame_index'],
                                    'cameras': {},
                                    'json': json_info['filepath'],
                                    'json_data': json_info['data']
                                }
                            
                            # Add camera view
                            frames[frame_key]['cameras'][camera_view] = str(filepath)
        
        print(f"Collected {len(frames)} episode-frame combinations")
        return frames

    def _apply_filters_and_generate_groups(self):
        """Apply filtering and generate frame groups for iteration."""
        if not self._extracted_frames:
            self._frame_groups = []
            return
        
        # Filter frames based on task filtering
        filtered_frames = self._extracted_frames
        
        if self.task_ids is not None:
            filtered_frames = {
                k: v for k, v in filtered_frames.items()
                if self._episodes_info.get(v['episode_index'], {}).get("task_index") in self.task_ids
            }
        
        # Generate frame groups
        self._frame_groups = []
        for frame_key, frame_data in filtered_frames.items():
            # Only create frame groups that have at least one requested camera view
            if frame_data['cameras']:
                frame_group = self._create_frame_group(frame_key, frame_data)
                if frame_group:  # Only add non-empty groups
                    self._frame_groups.append(frame_group)
        
        # Apply shuffling
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self._frame_groups)
        
        # Apply max_samples limit
        if self.max_samples is not None:
            self._frame_groups = self._frame_groups[:self.max_samples]
        
        print(f"Generated {len(self._frame_groups)} frame groups")

    def _sanitize_for_mongodb(self, value):
        """Sanitize a value to be MongoDB/FiftyOne compatible."""
        if value is None:
            return None
        elif isinstance(value, dict):
            # Recursively sanitize dictionaries, ensuring string keys
            sanitized_dict = {}
            for k, v in value.items():
                sanitized_key = str(k)
                sanitized_value = self._sanitize_for_mongodb(v)
                sanitized_dict[sanitized_key] = sanitized_value
            return sanitized_dict
        elif isinstance(value, (list, tuple)):
            # Recursively sanitize lists
            sanitized = []
            for item in value:
                sanitized.append(self._sanitize_for_mongodb(item))
            return sanitized
        elif hasattr(value, 'tolist'):  # numpy array
            return self._sanitize_for_mongodb(value.tolist())
        elif isinstance(value, (int, float, str, bool)):
            return value
        else:
            # Convert everything else to string
            return str(value)

    def _create_frame_group(self, frame_key: str, frame_data: Dict) -> List[fo.Sample]:
        """Create a frame group from extracted format data."""
        frame_group = []
        
        episode_idx = frame_data['episode_index']
        frame_idx = frame_data['frame_index']
        
        # Use cached JSON metadata if available
        json_metadata = {}
        if self.include_metadata and 'json_data' in frame_data:
            json_metadata = frame_data['json_data']
        elif frame_data['json'] and self.include_metadata:
            # Fallback: load from file if not cached
            try:
                with open(frame_data['json'], 'r') as f:
                    json_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load JSON metadata for {frame_key}: {e}")
        
        # Create a single group for this temporal frame
        group = fo.Group()
        
        # Create one sample per camera view
        for camera_view, image_path in frame_data['cameras'].items():
            if not os.path.exists(image_path):
                continue
            
            # Create sample with group element
            sample = fo.Sample(filepath=image_path, group=group.element(camera_view))
            
            # Add basic metadata (use JSON data if available, fallback to frame_data)
            sample["episode_index"] = json_metadata.get("episode_index", episode_idx)
            sample["frame_index"] = json_metadata.get("frame_index", frame_idx)
            sample["camera_view"] = camera_view
            
            # Add task information from episodes metadata
            if self._episodes_info and episode_idx in self._episodes_info:
                episode_info = self._episodes_info[episode_idx]
                task_idx = episode_info.get("task_index", 0)
                sample["task_index"] = task_idx
                
                if self._tasks_info and task_idx in self._tasks_info:
                    sample["task"] = self._tasks_info[task_idx].get("task", "")
            
            # Add trajectory data from JSON if available
            if json_metadata:
                # Core LeRobot fields (directly from JSON)
                if "timestamp" in json_metadata:
                    sample["timestamp"] = json_metadata["timestamp"]
                if "index" in json_metadata:
                    sample["global_index"] = json_metadata["index"]
                if "task_index" in json_metadata:
                    sample["task_index"] = json_metadata["task_index"]
                
                # Observation data (robot proprioception)
                if "observation.state" in json_metadata:
                    sample["observation_state"] = self._sanitize_for_mongodb(json_metadata["observation.state"])
                    
                # Action data (what the robot did)
                if "action" in json_metadata:
                    sample["action"] = self._sanitize_for_mongodb(json_metadata["action"])
                    
                # Additional observation modalities
                if "observation.velocity" in json_metadata:
                    sample["observation_velocity"] = self._sanitize_for_mongodb(json_metadata["observation.velocity"])
                    
                if "observation.effort" in json_metadata:
                    sample["observation_effort"] = self._sanitize_for_mongodb(json_metadata["observation.effort"])
                
                # Add other fields (flattened to avoid nested structures)
                excluded_fields = {
                    "timestamp", "observation.state", "action", "episode_index", 
                    "frame_index", "index", "task_index", "observation.velocity", 
                    "observation.effort"
                }
                
                for field, value in json_metadata.items():
                    if field not in excluded_fields:
                        # Replace dots with underscores for FiftyOne compatibility
                        field_name = field.replace(".", "_")
                        try:
                            # Use sanitization function for all values
                            sanitized_value = self._sanitize_for_mongodb(value)
                            sample[field_name] = sanitized_value
                        except Exception as e:
                            # Skip fields that can't be serialized
                            print(f"Warning: Could not serialize field '{field_name}': {e}")
                            pass
            
            # Set image metadata
            try:
                sample.metadata = fom.ImageMetadata.build_for(image_path)
            except:
                # Fallback to basic metadata
                sample.metadata = fom.ImageMetadata()
            
            frame_group.append(sample)
        
        return frame_group

    def __iter__(self):
        """Initialize iterator."""
        self._current_idx = 0
        return self

    def __len__(self):
        """Returns the total number of frame groups in the dataset."""
        if self._frame_groups is None:
            return 0
        return len(self._frame_groups)

    def __next__(self):
        """Returns the next sample in FiftyOne importer format."""
        # Initialize sample queue if needed
        if not hasattr(self, '_sample_queue'):
            self._sample_queue = []
            self._group_idx = 0
        
        # Fill queue with samples from next frame group if empty
        while not self._sample_queue and self._group_idx < len(self._frame_groups):
            frame_group = self._frame_groups[self._group_idx]
            self._sample_queue.extend(frame_group)
            self._group_idx += 1
        
        # Return next sample or raise StopIteration
        if not self._sample_queue:
            raise StopIteration
            
        sample = self._sample_queue.pop(0)
        
        # Return in FiftyOne importer format: (filepath, metadata, label)
        return sample.filepath, sample.metadata, None

    def get_dataset_info(self):
        """Returns a dict of information about the dataset."""
        info = {
            "type": "LeRobot Extracted Dataset",
            "format": "extracted",
            "total_frame_groups": len(self._frame_groups) if self._frame_groups else 0,
            "default_slice": str(self.default_slice),
            "group_field": str(self.group_field),
            "camera_views": [str(view) for view in self.camera_views],
        }
        
        # Add metadata from info.json (only safe, simple fields)
        if self._dataset_info:
            safe_fields = ["robot_type", "fps", "total_episodes", "total_frames", "total_tasks", "codebase_version"]
            for key, value in self._dataset_info.items():
                if key in safe_fields and not isinstance(value, (dict, list)):
                    # Only include simple types (str, int, float, bool)
                    if isinstance(value, (str, int, float, bool)):
                        info[str(key)] = value
        
        # Add episode count
        if self._episodes_info:
            info["episode_count"] = len(self._episodes_info)
        
        # Add task information (ensure keys are strings, values are simple)
        if self._tasks_info:
            tasks_dict = {}
            for task_idx, task_data in self._tasks_info.items():
                task_key = str(task_idx)
                task_value = str(task_data.get("task", ""))  # Convert to string
                tasks_dict[task_key] = task_value
            info["tasks"] = tasks_dict
        
        return info


# Register the importer with FiftyOne's type system
class LeRobotDatasetType(fo.Dataset):
    """FiftyOne dataset type for extracted LeRobot datasets."""
    
    def get_dataset_importer_cls(self):
        return LeRobotDatasetImporter
    
    @property
    def name(self):
        return "LeRobotDataset"


# Helper function to create LeRobot extracted grouped dataset
def create_lerobot_dataset(
    dataset_dir: str = None,
    data_path: str = None,
    labels_path: str = None,
    camera_views: List[str] = None,
    name: Optional[str] = None,
    episode_ids: Optional[List[int]] = None,
    task_ids: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
    default_slice: Optional[str] = None,
    include_metadata: bool = True,
    **kwargs
) -> fo.Dataset:
    """
    Create a FiftyOne grouped dataset from an extracted LeRobot dataset.
    
    Args:
        dataset_dir: Path to the root directory containing extracted_data/ and meta/
        data_path: Path to extracted_data directory (if different from dataset_dir/extracted_data)
        labels_path: Path to meta directory (if different from dataset_dir/meta)
        camera_views: List of camera view names (e.g., ["low", "high", "right_wrist"])
        name: Name for the FiftyOne dataset. If None, derived from directory name
        episode_ids: List of episode IDs to import. If None, imports all
        task_ids: List of task IDs to import. If None, imports all
        max_samples: Maximum number of frame groups to import
        shuffle: Whether to shuffle the import order
        seed: Random seed for shuffling
        default_slice: Default camera view slice. If None, uses first camera view
        include_metadata: Whether to load trajectory metadata from JSON files
        **kwargs: Additional arguments passed to the importer
        
    Returns:
        FiftyOne Dataset with grouped structure
    """
    if camera_views is None:
        raise ValueError(
            "camera_views is required. Please specify the camera view names "
            "(e.g., ['low', 'high', 'right_wrist', 'left_wrist'])"
        )
    
    # Create importer
    importer = LeRobotDatasetImporter(
        dataset_dir=dataset_dir,
        data_path=data_path,
        labels_path=labels_path,
        camera_views=camera_views,
        episode_ids=episode_ids,
        task_ids=task_ids,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
        default_slice=default_slice,
        include_metadata=include_metadata,
        **kwargs
    )
    
    # Create dataset name if not provided
    if name is None:
        if dataset_dir:
            name = f"lerobot-{Path(dataset_dir).name}"
        else:
            name = "lerobot-extracted"
    
    # Create dataset and add group field first
    dataset = fo.Dataset(name, overwrite=True)
    dataset.add_group_field(importer.group_field, default=importer.default_slice)
    
    # Setup the importer
    importer.setup()
    
    # Import all samples
    all_samples = []
    for frame_group in importer._frame_groups:
        all_samples.extend(frame_group)
    
    # Add samples to dataset
    if all_samples:
        dataset.add_samples(all_samples)
    
    # Add dataset info
    dataset.info.update(importer.get_dataset_info())
    
    print(f"Created LeRobot extracted dataset '{dataset.name}':")
    print(f"  - {len(dataset)} samples")
    print(f"  - {len(importer._frame_groups)} frame groups")
    print(f"  - Camera views: {importer.camera_views}")
    print(f"  - Default slice: {dataset.default_group_slice}")
    
    return dataset