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
import fiftyone.core.groups as fog
from fiftyone.utils.data.importers import LabeledImageDatasetImporter, ImportPathsMixin
import fiftyone.core.labels as fol
import fiftyone.core.metadata as fom
import fiftyone.core.utils as focu
import fiftyone.core.fields as fof
import fiftyone.types as fot


class LeRobotDatasetImporter(LabeledImageDatasetImporter, ImportPathsMixin):
    """
    Importer for extracted LeRobot format robotics datasets with grouped samples.
    
    Creates one group per frame with multiple camera views as slices.
    All group slices are created automatically during import.
    
    Expected structure:
    ```
    dataset_root/
    ├── extracted_data/
    │   ├── episode_000000/
    │   │   ├── episode_000000_000000_cam_low.png
    │   │   ├── episode_000000_000000_cam_high.png
    │   │   ├── episode_000000_000000.json
    │   │   └── ... (more frames)
    │   └── episode_000001/
    └── meta/
        ├── info.json
        ├── episodes.jsonl
        ├── tasks.jsonl
        └── stats.json
    ```
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
        # Validate that we have enough information to locate data
        if dataset_dir is None and data_path is None:
            raise ValueError(
                "Either 'dataset_dir' must be provided, or 'data_path' must be provided"
            )
        
        if camera_views is None or len(camera_views) == 0:
            raise ValueError(
                "camera_views is required and must be a non-empty list of camera view names "
                "(e.g., ['low', 'high', 'right_wrist', 'left_wrist'])"
            )
        
        # Call parent constructor first
        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )
        
        # Parse data path using ImportPathsMixin method
        self.data_path = self._parse_data_path(
            dataset_dir=dataset_dir,
            data_path=data_path,
            default="extracted_data",
        )
        
        # Parse labels path using ImportPathsMixin method  
        self.labels_path = self._parse_labels_path(
            dataset_dir=dataset_dir,
            labels_path=labels_path,
            default="meta",
        )
        
        # Store other parameters
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
        self._samples_list = None
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
        # Return a dict indicating we have custom fields including group field
        return {
            "episode_index": fof.IntField,
            "frame_index": fof.IntField,
            "camera_view": fof.StringField,
            "task_index": fof.IntField,
            "task": fof.StringField,
            "timestamp": fof.FloatField,
            "global_index": fof.IntField,
            "observation_state": fof.ListField,
            "action": fof.ListField,
            "observation_velocity": fof.ListField, 
            "observation_effort": fof.ListField,
            self.group_field: fof.EmbeddedDocumentField(fog.Group),
        }

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
        
        # Apply filtering and generate samples list (all camera views, all frames)
        self._apply_filters_and_generate_samples()
        
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
        """Parse filename to extract camera view for images."""
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
        """Collect all files organized by episode and frame using JSON metadata."""
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

    def _sanitize_for_mongodb(self, value):
        """Sanitize a value to be MongoDB/FiftyOne compatible."""
        if value is None:
            return None
        elif isinstance(value, dict):
            sanitized_dict = {}
            for k, v in value.items():
                sanitized_key = str(k)
                sanitized_value = self._sanitize_for_mongodb(v)
                sanitized_dict[sanitized_key] = sanitized_value
            return sanitized_dict
        elif isinstance(value, (list, tuple)):
            sanitized = []
            for item in value:
                sanitized.append(self._sanitize_for_mongodb(item))
            return sanitized
        elif hasattr(value, 'tolist'):  # numpy array
            return self._sanitize_for_mongodb(value.tolist())
        elif isinstance(value, (int, float, str, bool)):
            return value
        else:
            return str(value)

    def _apply_filters_and_generate_samples(self):
        """Apply filtering and generate samples list for iteration (ALL camera views for ALL frames)."""
        if not self._extracted_frames:
            self._samples_list = []
            return
        
        # Filter frames based on task filtering
        filtered_frames = self._extracted_frames
        
        if self.task_ids is not None:
            filtered_frames = {
                k: v for k, v in filtered_frames.items()
                if self._episodes_info.get(v['episode_index'], {}).get("task_index") in self.task_ids
            }
        
        # Generate samples for ALL camera views for ALL frames
        all_samples = []
        for frame_key, frame_data in filtered_frames.items():
            if frame_data['cameras']:
                frame_samples = self._create_all_camera_samples(frame_key, frame_data)
                all_samples.extend(frame_samples)
        
        # Apply shuffling
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(all_samples)
        
        # Apply max_samples limit
        if self.max_samples is not None:
            all_samples = all_samples[:self.max_samples]
        
        self._samples_list = all_samples
        print(f"Generated {len(self._samples_list)} samples across all camera views")

    def _create_all_camera_samples(self, frame_key: str, frame_data: Dict) -> List[Tuple]:
        """Create samples for ALL camera views for this frame (creating the full group)."""
        episode_idx = frame_data['episode_index']
        frame_idx = frame_data['frame_index']
        
        # Check if we have any of the requested camera views
        available_cameras = set(frame_data['cameras'].keys()) & set(self.camera_views)
        if not available_cameras:
            return []
        
        # Create a unique group ID for this frame
        group_id = focu.ObjectId()
        
        # Load JSON metadata once for all camera views
        json_metadata = {}
        if self.include_metadata and 'json_data' in frame_data:
            json_metadata = frame_data['json_data']
        elif frame_data['json'] and self.include_metadata:
            try:
                with open(frame_data['json'], 'r') as f:
                    json_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load JSON metadata for {frame_key}: {e}")
        
        # Create base label dict with common metadata
        base_label_dict = {
            "episode_index": json_metadata.get("episode_index", episode_idx),
            "frame_index": json_metadata.get("frame_index", frame_idx),
        }
        
        # Add task information from episodes metadata
        if self._episodes_info and episode_idx in self._episodes_info:
            episode_info = self._episodes_info[episode_idx]
            task_idx = episode_info.get("task_index", 0)
            base_label_dict["task_index"] = task_idx
            
            if self._tasks_info and task_idx in self._tasks_info:
                base_label_dict["task"] = self._tasks_info[task_idx].get("task", "")
        
        # Add trajectory data from JSON if available
        if json_metadata:
            # Core LeRobot fields
            if "timestamp" in json_metadata:
                base_label_dict["timestamp"] = json_metadata["timestamp"]
            if "index" in json_metadata:
                base_label_dict["global_index"] = json_metadata["index"]
            if "task_index" in json_metadata:
                base_label_dict["task_index"] = json_metadata["task_index"]
            
            # Observation data
            if "observation.state" in json_metadata:
                base_label_dict["observation_state"] = self._sanitize_for_mongodb(json_metadata["observation.state"])
                
            # Action data
            if "action" in json_metadata:
                base_label_dict["action"] = self._sanitize_for_mongodb(json_metadata["action"])
                
            # Additional observation modalities
            if "observation.velocity" in json_metadata:
                base_label_dict["observation_velocity"] = self._sanitize_for_mongodb(json_metadata["observation.velocity"])
                
            if "observation.effort" in json_metadata:
                base_label_dict["observation_effort"] = self._sanitize_for_mongodb(json_metadata["observation.effort"])
            
            # Add other fields (flattened)
            excluded_fields = {
                "timestamp", "observation.state", "action", "episode_index", 
                "frame_index", "index", "task_index", "observation.velocity", 
                "observation.effort"
            }
            
            for field, value in json_metadata.items():
                if field not in excluded_fields:
                    field_name = field.replace(".", "_")
                    try:
                        sanitized_value = self._sanitize_for_mongodb(value)
                        base_label_dict[field_name] = sanitized_value
                    except Exception as e:
                        print(f"Warning: Could not serialize field '{field_name}': {e}")
                        pass
        
        # Create one sample for each available camera view
        samples = []
        for camera_view in self.camera_views:  # Maintain consistent ordering
            if camera_view in available_cameras:
                image_path = frame_data['cameras'][camera_view]
                
                if not os.path.exists(image_path):
                    continue
                
                # Create image metadata for this camera
                try:
                    image_metadata = fom.ImageMetadata.build_for(image_path)
                except:
                    image_metadata = fom.ImageMetadata()
                
                # Create label dict for this specific camera view
                label_dict = base_label_dict.copy()
                label_dict["camera_view"] = camera_view
                label_dict[self.group_field] = fog.Group(name=camera_view, id=group_id)
                
                # Return tuple in FiftyOne importer format: (filepath, metadata, label)
                samples.append((image_path, image_metadata, label_dict))
        
        return samples

    def __iter__(self):
        """Initialize iterator."""
        self._current_idx = 0
        return self

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        if self._samples_list is None:
            return 0
        return len(self._samples_list)

    def __next__(self):
        """Returns the next sample in FiftyOne importer format."""
        if self._current_idx >= len(self._samples_list):
            raise StopIteration
            
        sample_tuple = self._samples_list[self._current_idx]
        self._current_idx += 1
        
        return sample_tuple  # Already in (filepath, metadata, label) format

    def get_dataset_info(self):
        """Returns a dict of information about the dataset."""
        info = {
            "type": "LeRobot Extracted Dataset",
            "format": "extracted_grouped",
            "total_samples": len(self._samples_list) if self._samples_list else 0,
            "default_slice": str(self.default_slice),
            "group_field": str(self.group_field),
            "camera_views": [str(view) for view in self.camera_views],
        }
        
        # Add metadata from info.json (only safe, simple fields)
        if self._dataset_info:
            safe_fields = ["robot_type", "fps", "total_episodes", "total_frames", "total_tasks", "codebase_version"]
            for key, value in self._dataset_info.items():
                if key in safe_fields and isinstance(value, (str, int, float, bool)):
                    info[str(key)] = value
        
        # Add episode count
        if self._episodes_info:
            info["episode_count"] = len(self._episodes_info)
        
        # Add task information
        if self._tasks_info:
            tasks_dict = {}
            for task_idx, task_data in self._tasks_info.items():
                task_key = str(task_idx)
                task_value = str(task_data.get("task", ""))
                tasks_dict[task_key] = task_value
            info["tasks"] = tasks_dict
        
        return info


# Dataset type class that automatically sets up grouped structure
class LeRobotDataset(fot.Dataset):
    """Dataset type for extracted LeRobot robotics datasets with grouped samples."""
    
    @property
    def name(self):
        """The name of the dataset type."""
        return "LeRobotDataset"
    
    def get_dataset_importer_cls(self):
        """Returns the dataset importer class."""
        return LeRobotDatasetImporter


# Override the from_dir method to automatically set up group slices
def _setup_grouped_dataset_after_import(dataset, camera_views, default_slice, group_field="group"):
    """Set up group slices and media types after import."""
    print("Setting up group slices...")
    
    # Add group slices for all camera views
    for camera_view in camera_views:
        try:
            dataset.add_group_slice(camera_view, "image")
        except ValueError:
            # Slice already exists
            pass
    
    # Set the default group slice
    if default_slice in camera_views:
        dataset.default_group_slice = default_slice
    
    print(f"Group setup complete. Available slices: {dataset.group_slices}")
    print(f"Default slice: {dataset.default_group_slice}")


# Monkey patch Dataset.from_dir to handle LeRobot datasets specially
_original_from_dir = fo.Dataset.from_dir

@classmethod
def _enhanced_from_dir(cls, dataset_type=None, **kwargs):
    """Enhanced from_dir that automatically sets up grouped datasets for LeRobot."""
    # Call the original from_dir method
    dataset = _original_from_dir(dataset_type=dataset_type, **kwargs)
    
    # If this is a LeRobot dataset, set up the group structure
    if dataset_type is LeRobotDataset or (hasattr(dataset_type, 'name') and dataset_type.name == "LeRobotDataset"):
        camera_views = kwargs.get('camera_views', [])
        default_slice = kwargs.get('default_slice', camera_views[0] if camera_views else None)
        group_field = kwargs.get('group_field', 'group')
        
        if camera_views:
            _setup_grouped_dataset_after_import(dataset, camera_views, default_slice, group_field)
    
    return dataset

# Apply the monkey patch
fo.Dataset.from_dir = _enhanced_from_dir


# Register the dataset type with FiftyOne's type system
fot.LeRobotDataset = LeRobotDataset