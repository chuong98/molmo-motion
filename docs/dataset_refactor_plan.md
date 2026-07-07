# Trajectory3DDataset Refactor Plan

## Context

`src/molmo_motion/data/trajectory_3d_dataset.py` is a single ~2360-line class that
composes **14 dataset variants** behind pervasive `if entry["_dataset"] == …`
dispatch (3D/2D track loaders, camera extrinsics, depth, video paths, captions,
point counts, benchmark entry builders) plus internal weighted mixing. It's hard
to read, extend, and test.

**Goal:** split it into a **base class + one subclass per dataset + a mixture
class**, each in its own file, so datasets are self-contained and a training run
selects/combines them cleanly.

**Decisions (locked):**
1. **Scope:** subclass the **7 core** training datasets (`egodex`, `ytvis`,
   `hepic`, `xperience`, `droid`, `stereo4d`, `molmospaces`) + the **3
   PointMotionBench** eval datasets (`hot3d_bench`, `worldtrack_bench`,
   `davis_bench`). **Drop** `egodex_hand`, `xperience_hand`, `davis`, `hotworld`
   — none are reachable via `get_dataset_by_name` tokens (`_HUMAN_DATASETS` +
   `_PER_DATASET_TOKENS`), so removing them changes no public behavior.
2. **Mixture:** a `TrajectoryMixtureDataset` holds a list of single-dataset
   instances and samples one by weight (sqrt/uniform/manual) then delegates —
   **replacing** the old internal `_ds_probs`/`_ds_to_indices` path (one mixing
   mechanism). `get_dataset_by_name` builds it, so training/eval entry points and
   the outer `IterableDatasetMixture` are unchanged.
3. **Fidelity:** behavior-preserving outputs (prompts/answers/metadata identical
   for the kept datasets); minor cleanups allowed (drop dead per-dataset code,
   turn `if ds in _SET` gates into overridable hooks / eligibility flags).

## Target module layout

New package `src/molmo_motion/data/trajectory/` (the old
`trajectory_3d_dataset.py` becomes a thin shim re-exporting for back-compat):

```
data/trajectory/
├── __init__.py           # re-export base, 10 subclasses, mixture, DATASET_REGISTRY, constants
├── base.py               # BaseTrajectoryDataset(Dataset): all shared logic + hook defaults
├── constants.py          # LABEL_TEXT_*, tokens, MOLMOSPACES_TIME_STRIDE, shared regexes
├── egodex.py             # EgoDexDataset
├── ytvis.py              # YTVisDataset
├── hdepic.py             # HdEpicDataset          (token "hepic")
├── xperience.py          # XperienceDataset
├── droid.py              # DroidDataset
├── stereo4d.py           # Stereo4DDataset
├── molmospaces.py        # MolmoSpacesDataset
├── hot3d_bench.py        # Hot3DBenchDataset
├── worldtrack_bench.py   # WorldTrackBenchDataset
├── davis_bench.py        # DavisBenchDataset
└── mixture.py            # TrajectoryMixtureDataset + build_from_tokens()
```

Back-compat shim: `data/trajectory_3d_dataset.py` keeps
`from molmo_motion.data.trajectory import *` and the old names
(`Trajectory3DDataset = BaseTrajectoryDataset`, `LABEL_TEXT_*`) so existing
imports (processor, full_rollout, tests) keep working.

## BaseTrajectoryDataset (base.py) — shared surface

Owns everything dataset-agnostic (verbatim moves from the monolith):

- **`__init__`**: all config flags (num_points, history_size, num_future_frames,
  mixed_history, use_camera_frame, use_depth_token, use_2d_point_features,
  use_2d_coordinate, v1_match_format, predict_history_3d, pred_end_point_first,
  eval_first_h_frames, depth_target_size, bspline_* ) + single-dataset entry
  assembly (split JSON load OR `_build_bench_entries`), `_filter_entries_by_npz_keys`,
  molmospaces stride compression hook, eval-config expansion, `data_root`.
  **No `datasets` tuple, no `_ds_probs`** — each instance is ONE dataset.
- **Item flow**: `__len__`, `get`, `_get_train` (samples entry uniformly from its
  own `self.entries`; no dataset-level sampling), `_get_eval`.
- **`_build_example`** (the big shared builder) + `_quantize`,
  `_format_tracks_text` (staticmethods), B-spline branch.
- **Shared helpers**: `_read_video_frames`, `_apply_depth_dropout`,
  `_unpack_3d_from_dict`, `_sample_2d`, `_w2c_from_pose_dict`,
  `_build_eval_configs`, `_build_eval_configs_first_h`, `_stratified_subsample`,
  `_get_object_names`.
- **Hook defaults** (overridden per subclass; defaults preserve current behavior
  for datasets that didn't support a feature):
  - `_load_3d_and_vis(entry, obj) -> (N,T,3),(N,T)` — abstract (must override)
  - `_load_2d_coords(...) -> (P,2)|None` — default `None`
  - `_load_w2c_for_frame(entry, t) -> (4,4)|None` — default `None`
  - `_load_depth_at_t(entry, t) -> (H,W)|None` — default `None`
  - `_get_video_path(entry) -> str|ndarray` — abstract
  - `_map_frame_to_video(entry, idxs) -> list` — default identity
  - `_get_point_count(entry, obj) -> int` — default 100
  - `_get_caption(entry) -> str` — default `entry.get("caption","")`
  - `_build_bench_entries() -> list` — default `[]`
  - Class attrs: `TOKEN`, `DATA_ROOT_ENV`, `DATA_ROOT_DEFAULT`, `SPLIT_FILE`,
    `SPLIT_IS_ABSOLUTE`, `IS_DICT_FORMAT` (bool), `DEPTH_TOKEN_ELIGIBLE` (bool),
    `TIME_STRIDE` (int, default 1).
- **Cleanup**: replace `if ds in _CAMERA_FRAME_SUPPORTED` → just call
  `self._load_w2c_for_frame` (None ⇒ skip). Replace `if ds in _DEPTH_SUPPORTED+…`
  → `if self.use_depth_token and self.DEPTH_TOKEN_ELIGIBLE`. Replace
  `if ds=="droid"` file-id suffix → overridable `_example_file_id(entry)`.
  Replace molmospaces stride branches → `self.TIME_STRIDE` used generically.

## Subclasses — what each overrides

Each subclass sets the class attrs + implements the hooks it needs (verbatim
extraction of the existing `_load_*_3d` / `_load_*_2d` / per-dataset branches):

| Subclass | 3D | 2D | w2c | depth | video | stride | point_count | notes |
|---|---|---|---|---|---|---|---|---|
| EgoDex | ✓ | ✓ | c2w⁻¹ | zip/EXR | mp4 | 1 | 100 | caption from stem |
| YTVis | ✓ | ✓ | c2w⁻¹ | zip/EXR | mp4 | 1 | 100 | dict format |
| HdEpic | ✓ | ✓(offset) | c2w⁻¹ | zip/EXR | mp4 | 1 | 100 | dict; token "hepic" |
| Xperience | ✓ | ✓ | c2w⁻¹ | hdf5 | mp4 | 1 | 100 | per-obj npz, 512² |
| Droid | ✓(T,N→N,T) | ✓ | identity | h5 mm→m | mp4 | 1 | 100 | `_example_file_id` adds cam |
| Stereo4D | ✓ | ✓ | look-at | — | mp4 | 1 | 100 | depth-eligible→monocular |
| MolmoSpaces | ✓(stride) | ✓(stride) | c2w⁻¹(raw) | npz mmap | mp4 | 4 | 100 | TIME_STRIDE=4 |
| Hot3DBench | ✓(2000) | ✓ | — | — | mp4 | 1 | 2000 | `_build_bench_entries` |
| WorldTrackBench | ✓(T,N→N,T) | project XYZ | — | — | jpeg bytes | 1 | 100 | split dirs |
| DavisBench | ✓(dict) | ✓ | — | — | mp4 | 1 | 100 | first obj |

## TrajectoryMixtureDataset (mixture.py)

- `__init__(self, sub_datasets: list[BaseTrajectoryDataset], weighting="sqrt")`.
- Train: `get(item, rng)` → pick sub-dataset index by weight
  (`w_i = sqrt(len(sub.entries))` for "sqrt", `1` for "uniform", dict for manual;
  normalize), then `return sub.get(item, rng)` (sub samples its own entry).
- Eval: concatenate each sub's `eval_configs`; `__len__` = total; `get(item)`
  routes to the owning sub + local index (deterministic, order-preserving).
- Satisfies the `Dataset` interface (`__len__`, `get`, `download`) so
  `DeterministicDataset`/`IterableDatasetMixture` wrap it unchanged.
- `build_from_tokens(tokens, split, **kwargs)` maps tokens→subclasses via
  `DATASET_REGISTRY` and returns a mixture (or the single instance if one token).

## get_dataset.py changes

`get_dataset_by_name` keeps parsing the grammar, then calls
`build_from_tokens(datasets, _split, num_points=…, num_future_frames=…,
history_size=…, use_2d_point_features=…, use_2d_coordinate=…,
max_eval_per_dataset=…, dataset_weighting="sqrt", use_camera_frame=True,
bspline_n_ctrl=…)`. Unknown/dropped tokens (davis/hotworld/hand) are already
absent from the token list, so no grammar change is needed.

## Tasks (bite-sized, each ends green + committed)

1. **Scaffold + constants + base skeleton**: create package, `constants.py`,
   `base.py` with `__init__`/entry-assembly/sampling/`_build_example`/helpers and
   hook defaults (move shared code; leave `_load_3d_and_vis`/`_get_video_path`
   abstract). Back-compat shim re-exports `Trajectory3DDataset=BaseTrajectoryDataset`.
2. **Characterization test FIRST** (safety net): with a synthetic entry +
   monkeypatched `_load_3d_and_vis` and `_read_video_frames`, snapshot
   `get()` output (question/answer/metadata) from the **current monolith**; assert
   the refactored base reproduces it byte-for-byte (frame mode + bspline mode).
3. **7 core subclasses** (one commit each or grouped): extract each dataset's
   loaders verbatim into its file; unit-test entry assembly against the real
   split JSONs under `/data/molmo_motion_1m` (entries load; counts > 0).
4. **3 benchmark subclasses**: extract `_build_*_bench_entries` + loaders;
   entry-assembly test against `/data/point_motion_bench` (tracks are extracted
   there).
5. **TrajectoryMixtureDataset**: implement + test weighting parity (sqrt weights
   = sqrt(entry counts)) and eval-config concatenation/routing.
6. **Rewire `get_dataset.py`** to `build_from_tokens`; test `get_dataset_by_name`
   builds a mixture for `_human` and a single instance for one token; verify
   `sft.py get_training_mixture` path still resolves.
7. **Delete monolith body**, keep shim; run full bspline + dataset test suites +
   ruff; update CLAUDE.md / dataset docstrings.

## Verification

- **Characterization test** (Task 2) is the core guarantee: identical `get()`
  output pre/post refactor for a synthetic sample, in both frame and B-spline modes.
- **Entry-assembly tests** (Tasks 3–4) run against the real split files / bench
  dirs on `/data` (no videos needed — only JSON + `.npz` headers).
- **Mixture tests** (Task 5): weighting math + deterministic eval routing.
- **Construction tests** (Task 6): `get_dataset_by_name("trajectory_3d_human_p8_h3_f8")`
  returns a mixture of 5; single-token returns one instance; `_ck10` still threads through.
- Existing `tests/test_bspline*.py` must stay green (imports via the shim).
- `ruff check` on all new files; no new errors in touched files.
- Full end-to-end train/eval parity awaits real videos (tracked separately).
