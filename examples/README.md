# Examples

[`01_quickstart.py`](01_quickstart.py) is a single end-to-end script: it runs
the model on a bundled clip and renders the predicted trajectory as an MP4 — a
2D track drawn over the static `t_0` frame, one `magma`-gradient polyline per
point with a bright dot at the moving end.

```bash
pip install -e ".[viz]"          # viz extra = matplotlib + imageio[ffmpeg]
python examples/01_quickstart.py                 # -> davis_bmx_trees_2d.mp4
```

The visualization is taken straight from the prediction tensor:

```python
out = model.predict_trajectory(**inputs)          # out.future_3d: (P, F, 3) meters
render_trajectory_mp4(out.future_3d, t0_image=..., intrinsics=..., points_2d_at_t0=...)
```

## No GPU?

The 4B model needs a GPU. To render the MP4 without one, use the bundled
released-model predictions (`data/predictions_h3.jsonl`) — this runs the exact
same `render_trajectory_mp4` path on an identical `(P, F, 3)` array:

```bash
python examples/01_quickstart.py --from-prediction
```

## Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--example NAME` | `davis_bmx_trees` | Sub-directory of `data/` to run on (see below). |
| `--from-prediction` | off | Skip the model; read the bundled prediction. |
| `--pad` | off | Extend the canvas (black border) so track that projects off the frame stays visible (the bmx clip rides off the right edge). |
| `--output PATH` | `<example>_2d.mp4` | Output MP4 path. |
| `--future-horizon N` | `30` | Frames to predict (model path only). |

## Bundled clips (`data/`)

Each clip ships `frame_t{-2,-1,+0}.jpg`, `points_2d_at_t0.pt` `(P,2)`,
`points_3d_history.pt` `(H,P,3)`, `intrinsics_K.pt` `(3,3)`, `meta.json`, and a
`caption.txt`. The five DAVIS/EgoDex clips also have a precomputed prediction in
`predictions_h3.jsonl` (so `--from-prediction` works for them):

| Clip | Predicted? | Notes |
|---|---|---|
| `davis_bmx_trees` | ✅ | Reference clip; rides off-frame — try `--pad`. |
| `davis_car_turn` | ✅ | |
| `davis_flamingo` | ✅ | |
| `egodex_ball_base` | ✅ | Egocentric. |
| `egodex_clean_surface` | ✅ | Egocentric. |
| `molmospaces_pick_place` | — (run the model) | Ships the ground-truth future for an ADE sanity check. |

`predictions_h3.jsonl` is the same per-example JSONL the eval pipeline
(`src/molmo_motion/eval/full_rollout.py`) writes: each row carries
`pred_raw_combined` `(P, F, 3)` — absolute camera-frame XYZ in meters, identical
to `out.future_3d`.
