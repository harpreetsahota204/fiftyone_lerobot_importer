# LeRobot Dataset Importer for FiftyOne

A streamlined FiftyOne importer for LeRobot datasets that have been extracted into individual PNG images and JSON metadata files. Creates properly grouped datasets with temporal relationships preserved across multiple camera views.

## Installation

1. **Install FiftyOne**:
   ```bash
   pip install fiftyone
   ```

2. **Clone this repository**:
   ```bash
   git clone https://github.com/harpreetsahota204/fiftyone_lerobot_importer.git
   cd fiftyone_lerobot_importer
   ```

## Expected Dataset Structure

Your LeRobot dataset should be in the extracted format:

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

## Converting from Parquet Format

If your LeRobot dataset is in the standard parquet format (with `data/chunk-000/episode_*.parquet` files), you can convert it to the extracted format using the provided extraction script:

```bash
# Convert parquet files to extracted PNG/JSON format
python extract_from_parquet_parallel.py \
    --input-dir <path_to_parquet_files> \
    --output-dir <path_to_output_directory> \
    --workers <select_number_of_workers>

# This will create the required directory structure:
# extracted_data/
# ├── episode_000000/
# │   ├── episode_000000_000000_cam_high.png
# │   ├── episode_000000_000000_cam_low.png
# │   ├── episode_000000_000000.json
# │   └── ...
# └── episode_000001/
#     └── ...
```

**Extraction Script Options:**
- `--input-dir`: Directory containing `episode_*.parquet` files
- `--output-dir`: Where to save extracted PNG/JSON files
- `--workers`: Number of parallel workers (default: CPU count / 2)
- `--keep-parquet`: Keep original parquet files after extraction
- `--test-one`: Process only one episode for testing
- `--sequential`: Disable multiprocessing for debugging

**Note**: The extraction process requires the `meta/` directory to already exist with `info.json`, `episodes.jsonl`, `tasks.jsonl`, and `stats.json` files. These are typically created when you download or prepare the LeRobot dataset.

## Usage

```python
import fiftyone as fo
from lerobot_importer import LeRobotDatasetImporter, LeRobotDataset

dataset_dir = "/path/to/your/dataset"
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=LeRobotDataset,
    camera_views=["low", "high", "right_wrist", "left_wrist"],
    labels_path="./meta",
    name="my_lerobot_dataset",
    include_metadata=True,
    overwrite=True,
)

print(f"Created dataset with {len(dataset)} samples")
print(f"Camera views: {dataset.group_slices}")
print(f"Default view: {dataset.default_group_slice}")
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_dir` | str | Yes | Root directory containing extracted_data/ and meta/ |
| `dataset_type` | class | Yes | Must be `LeRobotDataset` |
| `camera_views` | List[str] | **Yes** | Camera view names (e.g., ["low", "high", "right_wrist"]) |
| `labels_path` | str | No | Path to meta directory (default: "./meta") |
| `name` | str | Yes | Name for the FiftyOne dataset |
| `include_metadata` | bool | No | Load trajectory data from JSON files (default: True) |
| `episode_ids` | List[int] | No | Specific episode IDs to load (None = all) |
| `task_ids` | List[int] | No | Specific task IDs to load (None = all) |
| `max_samples` | int | No | Maximum samples to load (None = all) |
| `shuffle` | bool | No | Shuffle loading order (default: False) |
| `seed` | int | No | Random seed for shuffling |
| `default_slice` | str | No | Default camera view (default: first camera) |
| `overwrite` | bool | No | Overwrite existing dataset (default: False) |

## Example Variations

### Load specific episodes only:
```python
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=LeRobotDataset,
    camera_views=["low", "high"],
    episode_ids=[0, 1, 2, 3, 4],  # First 5 episodes only
    name="first_five_episodes",
    overwrite=True,
)
```

### Load single camera view:
```python
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=LeRobotDataset,
    camera_views=["high"],  # Only high camera
    name="high_camera_only",
    max_samples=100,
    overwrite=True,
)
```

### Skip trajectory metadata for faster loading:
```python
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=LeRobotDataset,
    camera_views=["low", "high", "right_wrist", "left_wrist"],
    include_metadata=False,  # Skip JSON loading
    name="fast_loading",
    overwrite=True,
)
```

## Working with Grouped Datasets

The importer automatically creates grouped datasets where each group represents one temporal frame with multiple camera views:

```python
# Access different camera views
high_cam = dataset.select_group_slices("high")
wrist_cam = dataset.select_group_slices("right_wrist")

# Group by episode for navigation
episodes_view = dataset.group_by("episode_index", order_by="frame_index")
dataset.save_view("by_episodes", episodes_view)
```

## Sample Data Structure

Each sample contains trajectory and metadata fields:

```python
sample = dataset.first()

# Basic fields
print(sample.episode_index)    # Episode number
print(sample.frame_index)      # Frame number within episode
print(sample.camera_view)      # Camera view name
print(sample.task)            # Task name

# Trajectory data (if include_metadata=True)
print(sample.timestamp)           # Frame timestamp
print(sample.observation_state)   # Robot joint states
print(sample.action)              # Robot actions
print(sample.observation_velocity) # Joint velocities
print(sample.observation_effort)   # Joint efforts
```

## Visualization

Launch FiftyOne App to visualize your dataset:

```python
session = fo.launch_app(dataset)
```

Navigate between camera views using the group slices dropdown in the FiftyOne interface.

## Requirements

- **Extracted format only**: Works exclusively with PNG/JSON files (no parquet support)
- **Camera views required**: You must specify which camera views to load
- **JSON metadata**: Uses episode/frame indices from JSON files, not filename parsing
- **Standard structure**: Expects extracted_data/ and meta/ directories

## Notes

- The importer creates one FiftyOne sample per camera view per frame
- All samples from the same frame share a group ID for proper temporal grouping  
- JSON trajectory data is automatically sanitized for MongoDB compatibility
- Large datasets may take time to load due to file I/O and metadata processing