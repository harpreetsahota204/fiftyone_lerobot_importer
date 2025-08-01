# LeRobot Dataset Importer for FiftyOne

A streamlined, opinionated FiftyOne importer specifically designed for LeRobot datasets that have been extracted into individual PNG images and JSON metadata files.

## Features

- **Extracted format only**: No support for parquet/video files - works exclusively with PNG/JSON

- **JSON-based metadata**: Uses episode/frame indices directly from JSON files (not filename parsing)

- **Explicit camera views**: Requires users to specify which camera views to load

- **Separate metadata**: Uses existing meta/ directory for episode and task information

- **Grouped datasets**: Creates FiftyOne grouped datasets preserving temporal relationships

- **Trajectory metadata**: Loads robot state, actions, and other data from JSON files

- **Episode grouping**: Easy grouping and navigation by episodes

## Installation

```bash
pip install fiftyone
```

No additional dependencies required (no pandas/pyarrow needed).

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

## Expected Dataset Structure

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

## Quick Start

```python
import fiftyone as fo
from lerobot_importer import LeRobotDatasetImporter, LeRobotDatasetType, create_lerobot_dataset

dataset_dir = "aloha_pen_uncap_diverse"

# Method 1: Using the helper function (recommended)
dataset = create_lerobot_dataset(
    dataset_dir=dataset_dir,
    dataset_type=LeRobotDatasetType(),
    name="aloha_pen_uncap",
    # REQUIRED: Specify the camera views you want to load
    camera_views=["low", "high", "right_wrist", "left_wrist"],
    # Optional: Filter specific episodes
    # episode_ids=list(range(10)),  # Only first 10 episodes
    include_metadata=True,  # Load JSON trajectory data
    # max_samples=100,  # Limit frame groups
    default_slice="high",  # Default camera view
    # overwrite=True
)

print(f"Created dataset with {len(dataset)} samples")
print(f"Camera views: {dataset.group_slices}")
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_dir` | str | Yes* | Root directory containing extracted_data/ and meta/ |
| `data_path` | str | Yes* | Path to extracted_data directory |
| `labels_path` | str | Yes* | Path to meta directory |
| `camera_views` | List[str] | **Yes** | Camera view names (e.g., ["low", "high", "right_wrist"]) |
| `episode_ids` | List[int] | No | Specific episode IDs to load (None = all) |
| `task_ids` | List[int] | No | Specific task IDs to load (None = all) |
| `include_metadata` | bool | No | Load trajectory data from JSON files (default: True) |
| `max_samples` | int | No | Maximum frame groups to load (None = all) |
| `shuffle` | bool | No | Shuffle loading order (default: False) |
| `seed` | int | No | Random seed for shuffling |
| `default_slice` | str | No | Default camera view (default: first camera) |

*Either `dataset_dir` OR both `data_path` and `labels_path` must be provided.

## Usage Examples

### Basic Usage

```python
# Using dataset root directory
dataset = create_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    name="my_dataset",
    camera_views=["low", "high", "right_wrist", "left_wrist"]
)
```

### Explicit Paths

```python
# Using explicit paths
dataset = create_lerobot_dataset(
    dataset_dir=None,
    dataset_type=LeRobotDatasetType(),
    data_path="/path/to/extracted_data",
    labels_path="/path/to/meta",
    name="my_dataset",
    camera_views=["low", "high"]
)
```

### Advanced Usage - Group by episodes for better navigation

```python
from lerobot_importer import create_lerobot_dataset

dataset = create_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDatasetType(),
    name="my_dataset",
    camera_views=["low", "high"],
    episode_ids=[0, 1, 2, 3, 4],  # First 5 episodes
    max_samples=100,
    include_metadata=True
)

# Group by episodes for better navigation
view = dataset.group_by("episode_index", order_by="frame_index")
dataset.save_view("episodes", view)
```

### Filtering Options

```python
# Single camera view
dataset = create_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDatasetType(),
    name="single_camera",
    camera_views=["high"],  # Only high camera
    max_samples=50
)

# Specific episodes
dataset = create_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDatasetType(),
    name="specific_episodes",
    camera_views=["low", "high"],
    episode_ids=[100, 101, 102],  # Only these episodes
)

# No metadata (faster loading)
dataset = create_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDatasetType(),
    name="no_metadata",
    camera_views=["high", "right_wrist"],
    include_metadata=False  # Skip JSON loading
)
```

## Working with Grouped Datasets

The importer creates grouped datasets where each group represents one temporal frame with multiple camera views:

```python
# Access different camera views
high_cam = dataset.select_group_slices("high")
wrist_cam = dataset.select_group_slices("right_wrist")

# Get all camera views for a specific frame
first_sample = dataset.first()
if first_sample.group:
    frame_group = dataset.select_groups([first_sample.group.id])
    print(f"Frame has {len(frame_group)} camera views")

# Group by episode for easy navigation
view = dataset.group_by("episode_index", order_by="frame_index")

# Save the view for easy loading in the App 
dataset.save_view("episodes", view)
```

## Sample Data Structure

Each sample contains:

```python
sample = dataset.first()

# Basic fields
print(sample.episode_index)    # Episode number
print(sample.frame_index)      # Frame number within episode
print(sample.camera_view)      # Camera view name
print(sample.task)            # Task name from meta/tasks.jsonl

# Trajectory data (if include_metadata=True)
print(sample.timestamp)           # Frame timestamp
print(sample.observation_state)   # Robot joint states
print(sample.action)              # Robot actions
print(sample.observation_velocity) # Joint velocities
print(sample.observation_effort)   # Joint efforts
```

## Filename Convention

The importer expects specific filename patterns:

- **Images**: `episode_XXXXXX_YYYYYY_cam_CAMERA.png`
  - `XXXXXX`: Episode index (6 digits, for matching only)
  - `YYYYYY`: Frame index (6 digits, for matching only)
  - `CAMERA`: Camera view name
- **JSON**: `episode_XXXXXX_YYYYYY.json`
  - Contains trajectory metadata including actual `episode_index` and `frame_index`

**Note**: The importer uses the `episode_index` and `frame_index` values directly from the JSON files, not from filename parsing. Filenames are only used for matching images to their corresponding JSON metadata.

## Error Handling

Common errors and solutions:

1. **"camera_views is required"**: You must specify which camera views to load

2. **"Data path does not exist"**: Check that extracted_data/ directory exists

3. **"Labels path does not exist"**: Check that meta/ directory exists

4. **"No episode directories found"**: Check that extracted_data/ contains episode_XXXXXX/ directories

5. **"No PNG files found"**: Check that episode directories contain properly named PNG files

## Testing Your Dataset

Test directly in Python:

```python
from lerobot_importer import create_lerobot_dataset

# Quick test with minimal data
test_dataset = create_lerobot_dataset(
    dataset_dir="/path/to/your/dataset",
    dataset_type=LeRobotDatasetType(),
    name="test_dataset",
    camera_views=["low", "high"],  # Adjust to your cameras
    episode_ids=[0],  # Just first episode
    max_samples=5,    # Just 5 frame groups
    include_metadata=True
)

print(f"✓ Success! {len(test_dataset)} samples loaded")
test_dataset.delete()  # Clean up
```

## Alternative Usage Methods

### Method 2: Direct Importer Usage

```python
from lerobot_importer import LeRobotDatasetImporter

# Create importer directly
importer = LeRobotDatasetImporter(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDatasetType(),
    camera_views=["low", "high", "right_wrist", "left_wrist"],
    include_metadata=True
)

# Create dataset manually
dataset = fo.Dataset("my_dataset", overwrite=True)
dataset.add_group_field("group", default="high")
importer.setup()

# Add all samples
all_samples = []
for frame_group in importer._frame_groups:
    all_samples.extend(frame_group)
dataset.add_samples(all_samples)
```

## Key Features

This importer is specifically designed for extracted datasets with these key features:

- **JSON-first**: Uses `episode_index` and `frame_index` directly from JSON files (not filename parsing)

- **No parquet support**: Only works with extracted PNG/JSON files

- **Explicit camera views**: Must specify camera view names (no auto-detection)

- **Separate metadata**: Uses existing meta/ directory structure

- **MongoDB-safe**: All data is properly sanitized for FiftyOne's database storage

- **Episode navigation**: Easy grouping and filtering by episodes

## Visualization

Launch FiftyOne App to visualize your dataset:

```python
session = fo.launch_app(dataset)
```

Navigate between camera views using the group slices dropdown in the FiftyOne interface.