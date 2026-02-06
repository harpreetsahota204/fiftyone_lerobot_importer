# LeRobot v3.0 Dataset Importer for FiftyOne

A FiftyOne importer for [LeRobot v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) format robotics datasets. Creates grouped video datasets where each group represents an episode with multiple camera views.

## Features

- **Native v3.0 Support**: Directly imports LeRobot v3.0 sharded datasets (no extraction needed)
- **Grouped Video Samples**: Each episode is a group with camera views as slices
- **Dynamic Schema**: All frame fields automatically parsed from `info.json` features — no hardcoded field names
- **Field Filtering**: Include/exclude fields with glob patterns (e.g., `exclude_fields=["*.is_fresh"]`)
- **Round-Trip Metadata**: Field mappings, semantic names, and normalization stats preserved for export back to LeRobot format
- **Browser Compatible**: All videos re-encoded to H.264/yuv420p for FiftyOne App playback
- **Auto-Detection**: Automatically detects camera views from dataset features, including multi-segment names

## Installation

1. **Install dependencies**:
   ```bash
   pip install fiftyone ffmpeg-python pyarrow pandas numpy
   ```

2. **Install ffmpeg with AV1 support** (required for video processing):
   
   LeRobot v3.0 datasets often use **AV1 codec** for video compression. You need ffmpeg with AV1 decoder support.
   
   ```bash
   # Ubuntu/Debian (recommended - includes AV1 support)
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Verify AV1 support
   ffmpeg -decoders | grep av1
   # Should show: libdav1d, libaom-av1, or av1
   ```
   
   **Note**: If using conda, the conda ffmpeg may lack AV1 support. Remove it to use system ffmpeg:
   ```bash
   conda remove --force ffmpeg
   ```

3. **Clone this repository**:
   ```bash
   git clone https://github.com/harpreetsahota204/fiftyone_lerobot_importer.git
   cd fiftyone_lerobot_importer
   ```

## Expected Dataset Structure (LeRobot v3.0)

```
dataset_root/
├── meta/
│   ├── info.json              # Schema, FPS, codebase_version
│   ├── stats.json             # Global feature statistics
│   ├── episodes/              # Chunked Parquet episode metadata
│   │   └── chunk-000/
│   │       └── episodes_000.parquet
│   └── tasks/                 # Chunked Parquet task descriptions
│       └── chunk-000/
│           └── file_000.parquet
├── data/                      # Sharded Parquet data files
│   └── chunk-000/
│       └── file-000.parquet   # Contains multiple episodes
└── videos/                    # Sharded MP4 videos per camera
    ├── cam_high/
    │   └── chunk-000/
    │       └── file-000.mp4   # Contains multiple episodes
    └── cam_low/
        └── chunk-000/
            └── file-000.mp4
```

## Quick Start

```python
import fiftyone as fo
from lerobot_importer import import_lerobot_dataset

# Import dataset with a single function call
dataset = import_lerobot_dataset(
    dataset_dir="/path/to/lerobot_dataset",
    name="my_robot_data",
)

# Launch FiftyOne App
session = fo.launch_app(dataset)
```

## Usage

### Basic Import

```python
import fiftyone as fo
from lerobot_importer import LeRobotDatasetImporter

# Create importer
importer = LeRobotDatasetImporter(
    dataset_dir="/path/to/lerobot_dataset",
    camera_views=["cam_high", "cam_low"],  # or None for auto-detect
)

# Create dataset and import
dataset = fo.Dataset("my_robot_dataset")
importer.import_to_dataset(dataset)

print(f"Imported {len(dataset)} samples")
print(f"Camera views: {dataset.group_slices}")
```

### Import with Episode Filtering

```python
# Import specific episodes
dataset = import_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    episode_ids=[0, 1, 2, 3, 4],  # First 5 episodes
    max_samples=10,               # Or limit total episodes
)

# Filter by task
dataset = import_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    task_ids=[0],  # Only episodes with task_index=0
)
```

### Import with Field Filtering

```python
# Only import observation and action fields (skip freshness flags, etc.)
dataset = import_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    include_fields=["observation.*", "action.*", "timestamp"],
)

# Import everything except freshness flags
dataset = import_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    exclude_fields=["*.is_fresh"],
)

# Combine: only observation/action fields, minus freshness
dataset = import_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    include_fields=["observation.*", "action.*", "timestamp"],
    exclude_fields=["*.is_fresh"],
)
```

### Skip Frame-Level Data (Faster Import)

```python
dataset = import_lerobot_dataset(
    dataset_dir="/path/to/dataset",
    include_frame_data=False,  # Skip loading states/actions per frame
)
```

### Custom Clips Directory

```python
importer = LeRobotDatasetImporter(
    dataset_dir="/path/to/dataset",
    clips_dir="/path/to/custom/clips",  # Where to save extracted episode clips
    overwrite_clips=True,  # Re-extract even if clips exist
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_dir` | str/Path | **Required** | Root directory of v3.0 dataset |
| `camera_views` | List[str] | None | Camera views to import (None = auto-detect) |
| `episode_ids` | List[int] | None | Specific episodes to import (None = all) |
| `task_ids` | List[int] | None | Filter by task IDs (None = all) |
| `clips_dir` | str/Path | `{dataset_dir}/episode_clips` | Directory for extracted clips |
| `include_frame_data` | bool | True | Load frame-level states/actions |
| `include_fields` | List[str] | None | Glob patterns for fields to include (None = all) |
| `exclude_fields` | List[str] | None | Glob patterns for fields to exclude (None = none) |
| `max_samples` | int | None | Maximum episodes to import |
| `shuffle` | bool | False | Shuffle episode order |
| `seed` | int | None | Random seed for shuffling |
| `default_slice` | str | First camera | Default camera view |
| `group_field` | str | "group" | Name of group field |
| `overwrite_clips` | bool | False | Re-extract existing clips |

## Working with the Dataset

### Access Grouped Samples

```python
# Iterate over episode groups
for group in dataset.iter_groups():
    episode_idx = group.first().episode_index
    print(f"Episode {episode_idx}:")
    for sample in group.values():
        print(f"  Camera: {sample.camera_view}")

# Select specific camera view
high_cam_view = dataset.select_group_slices("cam_high")
```

### Access Frame-Level Data

Frame fields are dynamically imported from `info.json` features. The LeRobot dot-notation
names are flattened with underscores (e.g., `observation.state` becomes `observation_state`).

```python
sample = dataset.first()

# Access frames (1-indexed in FiftyOne)
frame_1 = sample.frames[1]
print(frame_1.observation_state)  # list: Robot joint states
print(frame_1.action)             # list: Robot actions (or action_absolute, action_relative, etc.)
print(frame_1.timestamp)          # float: Frame timestamp

# Complex datasets may have many more fields, e.g.:
# frame_1.observation_wrist_wrench       # list: Force/torque sensor
# frame_1.observation_end_effector_pose_absolute  # list: 6DOF pose
# frame_1.action_arm                     # list: Arm joint subset
# frame_1.next_done                      # bool: Episode boundary flag

# Get all frame data
for frame_num, frame in sample.frames.items():
    print(f"Frame {frame_num}: state={frame.observation_state}")
```

### Convert to Frame Dataset

```python
# Convert videos to individual frame samples
frames_dataset = dataset.to_frames()

# Now each sample is a single frame
print(len(frames_dataset))  # Total number of frames
```

### Filter and Query

```python
# Filter by task
task_view = dataset.match(F("task") == "Pick up the cube")

# Filter by episode
episode_view = dataset.match(F("episode_index") == 5)

# Sort by episode
sorted_view = dataset.sort_by("episode_index")
```

## Sample Data Structure

Each video sample contains:

```python
sample = dataset.first()

# Episode-level fields
sample.episode_index       # int: Episode number
sample.camera_view         # str: Camera name (e.g., "hand", "top")
sample.task_index          # int: Task ID
sample.task                # str: Task description
sample.dataset_from_index  # int: Start row in source parquet
sample.dataset_to_index    # int: End row in source parquet
sample.group               # Group: Links samples from same episode

# Frame-level fields are dynamic — determined by info.json features.
# Every non-video feature is imported with dots replaced by underscores.
# Scalar fields (shape [1]) become Python primitives (float, int, bool).
# Array fields (shape [N]) become Python lists.
sample.frames[1].observation_state  # list[float]: Robot joint states
sample.frames[1].action             # list[float]: Robot actions
sample.frames[1].timestamp          # float: Timestamp in seconds
```

### Dataset Info (Round-Trip Metadata)

All metadata needed to export back to LeRobot format is stored in `dataset.info`:

```python
dataset.info["lerobot_field_map"]   # {"observation_state": "observation.state", ...}
dataset.info["field_names"]         # {"observation_state": ["arm_lift_joint", ...]}
dataset.info["video_feature_map"]   # {"hand": "observation.image.hand", ...}
dataset.info["features"]            # Full info.json features dict
dataset.info["stats"]               # Normalization statistics (for ML training)
dataset.info["tasks"]               # Task vocabulary mapping
dataset.info["fps"]                 # Frames per second
dataset.info["robot_type"]          # Robot type string
```

## How It Works

1. **Validates** the dataset is v3.0 format (checks `meta/info.json`)
2. **Loads metadata** from chunked Parquet files in `meta/episodes/` and `meta/tasks/`
3. **Builds frame field schema** dynamically from `info.json` features, applying include/exclude filters
4. **Extracts episode clips** from sharded MP4 videos using timestamps from episode metadata
5. **Re-encodes** clips to H.264/yuv420p for browser compatibility (stream-copies when source is already H.264)
6. **Loads frame data** from sharded Parquet files in `data/`, converting types per the schema
7. **Creates grouped samples** where each episode is a group with camera slices

### Dynamic Schema

The importer reads `info.json` features and maps each field type to FiftyOne:

| LeRobot dtype | Shape | FiftyOne type | Python type |
|---------------|-------|---------------|-------------|
| `video` | any | Skip (group slices) | — |
| `float32` / `float64` | `[1]` | `FloatField` | `float` |
| `float32` / `float64` | `[N]` | `ListField` | `list[float]` |
| `int64` | `[1]` | `IntField` | `int` |
| `bool` | `[1]` | `BooleanField` | `bool` |
| `bool` / `int64` | `[N]` | `ListField` | `list` |

Field names are converted from dot-notation to underscores: `observation.state` becomes `observation_state`.

## Requirements

- **Python 3.8+**
- **FiftyOne 0.23.0+**
- **ffmpeg** (system installation)
- **LeRobot v3.0 format** datasets only

## Notes

- First import extracts video clips (may take time); subsequent imports reuse cached clips
- All videos are re-encoded to ensure FiftyOne App compatibility
- Frame-level data is loaded from Parquet files and stored in `sample.frames`
- Large datasets benefit from `max_samples` for initial testing

## Troubleshooting

### "This importer only supports LeRobot v3.0 format"
Your dataset is in an older format. Use LeRobot's conversion tools:
```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=your/dataset
```

### "ffmpeg is not installed"
Install ffmpeg on your system (see Installation section).

### "Decoder (codec av1) not found"
Your ffmpeg doesn't support AV1 decoding. LeRobot v3.0 uses AV1 codec.

**Solution**: Use system ffmpeg instead of conda ffmpeg:
```bash
# Remove conda ffmpeg
conda remove --force ffmpeg

# Verify system ffmpeg has AV1 support
ffmpeg -decoders | grep av1
```

### Videos don't play in FiftyOne App
The importer automatically re-encodes videos to H.264/yuv420p. If issues persist:
```python
import fiftyone.utils.video as fouv
fouv.reencode_videos(dataset)
```

## License

Apache 2.0
