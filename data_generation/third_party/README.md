# Third-party models

The pipeline composes three frozen, publicly-released models. Each is vendored
here with its upstream license retained. Only the thin integration scripts we add
on top are ours.

| Dir | Upstream | Role | License |
|-----|----------|------|---------|
| `sam3/` | [SAM 3](https://huggingface.co/facebook/sam3) (Meta) | object segmentation from a point prompt | see `sam3/LICENSE` |
| `alltracker/` | [AllTracker](https://huggingface.co/aharley/alltracker) (Harley et al.) | dense 2D point tracking | see `alltracker/LICENSE` |
| `vipe/` | [ViPE](https://github.com/nvlabs/vipe) (NVIDIA) | metric depth + camera pose | see `vipe/LICENSE`, `vipe/THIRD_PARTY_LICENSES.md` |

Grounding language models are pulled from HuggingFace at runtime (not vendored):
`allenai/MolmoPoint-Vid-4B`, `allenai/Molmo2-8B`, `Qwen/Qwen3-0.6B`.

## Our integration scripts (the only modified/added files)

- `sam3/molmo2_pointing.py` — MolmoPoint pointing, Qwen3 object-phrase extraction,
  Molmo2-8B re-captioning, vague-phrase detection.
- `sam3/querypoints_from_video.py` — SAM 3 segmentation + K-means query-point sampling.
- `alltracker/run-query-points.py` — AllTracker inference on a set of query points.
- `vipe/scripts/vipe_to_colmap_general.py` — back-projection of 2D tracks to a metric
  3D world frame (Stage 4).
- `vipe/track-filter-smooth.py` — consensus-gated trust weighting + ray-only
  smoothing (Stage 5).

## Build notes

- **SAM 3 / ViPE**: `pip install -e third_party/sam3` and `pip install -e third_party/vipe`.
  ViPE compiles a CUDA extension on install — it needs `nvcc` (CUDA 12.x) and `ninja`.
  On first build it downloads Eigen automatically.
- **AllTracker**: used as plain source (no install). Its checkpoint downloads from
  `aharley/alltracker` via `torch.hub` on first use.
