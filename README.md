# LeRobot v3.0 Dataset Importer for FiftyOne

A FiftyOne importer for [LeRobot v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) format robotics datasets. Creates grouped video datasets where each group represents an episode with multiple camera views.

## Features

- **Grouped Video Samples**: Each episode is a group with camera views as slices
- **Dynamic Schema**: All frame fields automatically parsed from `info.json` — no hardcoded field names
- **Field Filtering**: Include/exclude fields with glob patterns (e.g., `exclude_fields=["*.is_fresh"]`)
- **Auto-Detection**: Automatically detects camera views from dataset features
- **Browser Compatible**: Videos re-encoded to H.264/yuv420p for FiftyOne App playback

## Installation

1. **Install dependencies**:
   ```bash
   pip install fiftyone ffmpeg-python pyarrow tqdm
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

## Quick Start

### Download a LeRobot v3.0 Dataset

Datasets must be in [LeRobot v3.0 format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3). You can download one from the Hugging Face Hub:

```python
from huggingface_hub import snapshot_download

dataset_path = snapshot_download(
    repo_id="your-org/your-dataset",  # e.g., "lerobot/aloha_sim_insertion_human"
    repo_type="dataset",
    local_dir="my_dataset",
)
```

### Import into FiftyOne

```python
import fiftyone as fo
from lerobot_importer import LeRobotDataset

dataset = fo.Dataset.from_dir(
    dataset_dir="my_dataset",
    dataset_type=LeRobotDataset,
    name="my_robot_data",
)

# Launch FiftyOne App
session = fo.launch_app(dataset)
```

## Usage

### Import with Episode Filtering

```python
# Import specific episodes
dataset = fo.Dataset.from_dir(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDataset,
    name="my_dataset",
    episode_ids=[0, 1, 2, 3, 4],
    max_samples=10,
)

# Filter by task
dataset = fo.Dataset.from_dir(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDataset,
    name="my_dataset",
    task_ids=[0],
)
```

### Import with Field Filtering

```python
# Only import observation and action fields
dataset = fo.Dataset.from_dir(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDataset,
    name="my_dataset",
    include_fields=["observation.*", "action.*", "timestamp"],
)

# Import everything except freshness flags
dataset = fo.Dataset.from_dir(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDataset,
    name="my_dataset",
    exclude_fields=["*.is_fresh"],
)

# Combine both
dataset = fo.Dataset.from_dir(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDataset,
    name="my_dataset",
    include_fields=["observation.*", "action.*", "timestamp"],
    exclude_fields=["*.is_fresh"],
)
```

### Skip Frame-Level Data (Faster Import)

```python
dataset = fo.Dataset.from_dir(
    dataset_dir="/path/to/dataset",
    dataset_type=LeRobotDataset,
    name="my_dataset",
    include_frame_data=False,
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
| `overwrite_clips` | bool | False | Re-extract existing clips |

## Working with the Dataset

### Sample Structure

Each video sample has episode-level fields and frame-level data:

```python
sample = dataset.first()

# Episode-level fields
sample.episode_index       # int: Episode number
sample.camera_view         # str: Camera name (e.g., "hand", "top")
sample.task                # str: Task description
sample.task_index          # int: Task ID

# Frame-level fields (1-indexed)
# Field names match info.json with dots replaced by underscores
sample.frames[1].observation_state  # list[float]: Robot joint states
sample.frames[1].action             # list[float]: Robot actions
sample.frames[1].timestamp          # float: Timestamp in seconds
```

### Field Metadata

Each frame field carries its LeRobot metadata via `description` and `info`:

```python
field = dataset.get_field("frames.observation_state")
field.description  # "[arm_lift_joint, arm_flex_joint, ...]"
field.info         # {"lerobot_name": "observation.state", "dtype": "float32", "shape": [8], "names": [...]}
```

### Grouped Samples

```python
# Iterate over episode groups
for group in dataset.iter_groups():
    for sample in group.values():
        print(f"Episode {sample.episode_index}, Camera: {sample.camera_view}")

# Select a specific camera view
hand_view = dataset.select_group_slices("hand")
```

### Filter and Query

```python
from fiftyone import ViewField as F

task_view = dataset.match(F("task") == "Pick up the cube")
episode_view = dataset.match(F("episode_index") == 5)
sorted_view = dataset.sort_by("episode_index")
```

### Dataset Info

Dataset-level metadata is stored in `dataset.info`:

```python
dataset.info["features"]            # Full info.json features dict
dataset.info["video_feature_map"]   # {"hand": "observation.image.hand", ...}
dataset.info["stats"]               # Normalization statistics
dataset.info["tasks"]               # Task vocabulary mapping
dataset.info["fps"]                 # Frames per second
dataset.info["robot_type"]          # Robot type string
```

## Troubleshooting

### "This importer only supports LeRobot v3.0 format"
Your dataset is in an older format. Use LeRobot's conversion tools:
```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=your/dataset
```

### "ffmpeg is not installed"
Install ffmpeg on your system (see Installation section).

### "Decoder (codec av1) not found"
Your ffmpeg doesn't support AV1 decoding. Use system ffmpeg instead of conda ffmpeg:
```bash
conda remove --force ffmpeg
ffmpeg -decoders | grep av1
```

### Videos don't play in FiftyOne App
The importer re-encodes videos to H.264/yuv420p automatically. If issues persist:
```python
import fiftyone.utils.video as fouv
fouv.reencode_videos(dataset)
```

## License

Apache 2.0
