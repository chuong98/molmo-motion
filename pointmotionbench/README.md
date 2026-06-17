# PointMotionBench — dataset construction

PointMotionBench is a benchmark for evaluating 3D point motion in video, covering egocentric and third-person scenes across three source datasets. Each sample pairs an RGB video clip with per-object 3D and 2D tracked surface points and a human-verified natural-language caption.

The benchmark data lives on HuggingFace:

**[allenai/PointMotionBench](https://huggingface.co/datasets/allenai/PointMotionBench)**

Tracks and annotations for all three sub-datasets are hosted there; videos must be reconstructed from their upstream sources.

Download:

```bash
huggingface-cli download allenai/PointMotionBench \
    --repo-type dataset --local-dir $POINTMOTIONBENCH_ROOT
```

Then follow the [allenai/PointMotionBench](https://huggingface.co/datasets/allenai/PointMotionBench) README to reconstruct the videos for each sub-dataset.
