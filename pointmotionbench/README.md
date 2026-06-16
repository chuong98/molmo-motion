# PointMotionBench

A benchmark for evaluating 3D point motion in video, covering egocentric and third-person scenes across three source datasets. Each sample pairs an RGB video clip with per-object 3D and 2D tracked surface points and a human-verified natural-language caption.

## Overview

| Dataset | Clips | Video format | Tracks | Scene type |
|---------|-------|--------------|--------|------------|
| DAVIS | 90 | mp4, 24 fps | 2D + 3D | Third-person, diverse outdoor/indoor |
| HOT3D | 2,475 | mp4, 30 fps | 2D + 3D | Egocentric, object manipulation (Aria) |
| WorldTrack | 155 | npz (frames embedded), 30 fps | 3D (+2D) | Egocentric + studio, 4 splits |

---

## Setup

### Step 1 — Download from HuggingFace

Tracks, videos, and annotations are hosted at [allenai/PointMotionBench](https://huggingface.co/datasets/allenai/PointMotionBench) (HuggingFace).

Download:

```python
# pip install huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="allenai/PointMotionBench",
    repo_type="dataset",
    local_dir=".",           # downloads davis/, hot3d/, and worldtrack/ into the current directory
)
```

Run this from (or set `local_dir` to) the directory you export as
`POINTMOTIONBENCH_ROOT` for evaluation — see the
[top-level README](../README.md#downloading-the-dataset-and-benchmark).
The reconstruction commands below assume you run them from that same
directory, so outputs land in `hot3d/videos/` and `worldtrack/` next to
the downloaded annotations.

---

### Step 2 — HOT3D: Download Videos

We do not share videos from HOT3D. Users should access the videos from the source dataset at [bop-benchmark/hot3d](https://huggingface.co/datasets/bop-benchmark/hot3d) (HuggingFace).

**Requirements:** `imageio[ffmpeg]`, `imageio-ffmpeg`, `opencv-python-headless`, `numpy`

```bash
# 1. download train_aria TARs (~1,516 clips)
#    to download only the 1,272 clips needed for PointMotionBench, add:
#    --captions hot3d/hot3d_annotations.json
python hot3d/download_train_aria.py --output /path/to/train_aria

# 2. extract undistorted upright RGB videos (one mp4 per TAR)
python hot3d/extract_rgbs.py \
    --clips_dir  /path/to/train_aria \
    --output_dir /path/to/rgbs

# 3. trim to PointMotionBench windows
python hot3d/trim_hot3d_clips.py \
    --src_dir    /path/to/rgbs \
    --captions   hot3d/hot3d_annotations.json \
    --output_dir hot3d/videos
```

For large-scale extraction, `extract_rgbs.py` supports sharding:

```bash
python hot3d/extract_rgbs.py \
    --clips_dir  /path/to/train_aria \
    --output_dir /path/to/rgbs \
    --shard_idx  0 \
    --num_shards 8
```

---

### Step 3 — WorldTrack: extract clips

Download the WorldTrack source data (WorldTrack benchmark, introduced in St4RTrack, Feng et al., ICCV 2025 — dataset download available at [HavenFeng/St4RTrack](https://github.com/HavenFeng/St4RTrack)). The source data should have this layout:

```
WorldTrack/
├── adt_mini/        # Aria Digital Twin
├── ds_mini/         # Dynamic Scenes
├── po_mini/         # POtential Objects
└── pstudio_mini/    # PStudio
```

Then extract PointMotionBench clips using the index map from Step 1:

```bash
python worldtrack/extract_worldtrack_clips.py \
    --index_map  worldtrack/worldtrack_index_map.json \
    --src_dir    /path/to/WorldTrack \
    --output_dir worldtrack
```

| Split | Clips | Frames per clip | Scene type |
|-------|-------|-----------------|------------|
| `adt_mini` | 39 | 12–300 | Apartment indoor, egocentric (Aria Digital Twin) |
| `ds_mini` | 52 | 39–128 | Dynamic indoor scenes |
| `po_mini` | 16 | 78–128 | Mixed indoor (cab, seminar, egobody) |
| `pstudio_mini` | 48 | 150 | Studio sports (basketball, football, tennis, etc.) |
