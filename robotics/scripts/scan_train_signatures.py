"""Scan a prepared training data view and dump the seen-house /
seen-pickup-object / seen-receptacle signatures to JSON.

Used by ``build_eval_subset.py`` to classify each benchmark episode into
``seen_house × seen_pickup`` buckets. ``Seen`` means: the entity (house,
pickup-object UUID, or receptacle UUID) appears in at least one training
trajectory.

Walks the layout written by ``prepare_training_data.py``:

    <src_root>/<scenario>/(train|val)/house_<N>/trajectories_batch_*.h5

Only ``train`` houses count as seen — val houses are held out from training
and should not be treated as training coverage.

Usage:

    python scripts/scan_train_signatures.py \\
        --src_root /path/to/train_view \\
        --scenarios pick_place_2cam_randomized pick_place_color_2cam_randomized \\
        --output train_signatures.json

Output JSON schema:

    {
      "scenarios": [...],
      "src_root": "...",
      "num_h5_files": <int>,
      "num_trajs_scanned": <int>,
      "houses": [<int>, ...],            # house indices that appear in train
      "pickup_uuids": ["<uuid>", ...],   # pickup-object UUIDs in train
      "receptacle_uuids": ["<uuid>", ...]
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py


log = logging.getLogger(__name__)


def _scan_h5(h5_path: Path) -> tuple[set[str], set[str], int]:
    """Return (pickup_uuids, receptacle_uuids, n_trajectories) from one h5."""
    pickup_uuids: set[str] = set()
    recep_uuids: set[str] = set()
    n = 0
    try:
        with h5py.File(h5_path, "r") as f:
            for tk in [k for k in f.keys() if k.startswith("traj_")]:
                n += 1
                try:
                    raw = f[tk]["obs_scene"][()]
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    elif hasattr(raw, "tobytes"):
                        raw = raw.tobytes().decode()
                    sc = json.loads(raw)
                    obj = sc.get("object_name") or ""
                    if "_" in obj:
                        pickup_uuids.add(obj.split("_")[1])
                    rec = sc.get("place_receptacle_name") or ""
                    if "/" in rec:
                        recep_uuids.add(rec.rsplit("/", 1)[-1])
                except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    log.debug(f"  {h5_path.name}:{tk} parse error: {e}")
    except (OSError, RuntimeError) as e:
        log.warning(f"  open failed {h5_path}: {e}")
    return pickup_uuids, recep_uuids, n


def _iter_house_dirs(scenario_root: Path):
    """Yield ``(split, house_dir)`` for every ``house_<N>/`` under a
    scenario root, where split ∈ {``train``, ``val``, ``root``}."""
    for split in ("train", "val"):
        sub = scenario_root / split
        if sub.is_dir():
            for d in sub.iterdir():
                if d.is_dir() and d.name.startswith("house_"):
                    yield split, d
    for d in scenario_root.iterdir():
        if d.is_dir() and d.name.startswith("house_"):
            yield "root", d


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src_root", type=Path, required=True,
        help="Prepared training view root (output of prepare_training_data.py).",
    )
    p.add_argument(
        "--scenarios", nargs="+",
        default=["pick_place_2cam_randomized", "pick_place_color_2cam_randomized"],
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--log_every", type=int, default=400,
        help="Log progress every N houses.",
    )
    args = p.parse_args()

    all_houses: set[int] = set()
    all_pickup: set[str] = set()
    all_recep: set[str] = set()
    n_trajs_total = 0
    n_h5 = 0

    for scenario in args.scenarios:
        root = args.src_root / scenario
        if not root.is_dir():
            log.error(f"missing scenario dir: {root}")
            continue

        houses = sorted(set(_iter_house_dirs(root)), key=lambda t: t[1].name)
        log.info(f"[{scenario}] {len(houses)} house dirs")
        t0 = time.time()
        for i, (split, house_dir) in enumerate(houses):
            try:
                house_idx = int(house_dir.name[len("house_"):])
            except ValueError:
                continue
            # Only `train`-side houses count as seen.
            if split == "val":
                continue
            all_houses.add(house_idx)
            for h5p in house_dir.glob("trajectories_batch_*.h5"):
                n_h5 += 1
                p_uids, r_uids, nt = _scan_h5(h5p)
                all_pickup |= p_uids
                all_recep |= r_uids
                n_trajs_total += nt
            if (i + 1) % args.log_every == 0 or (i + 1) == len(houses):
                dt = time.time() - t0
                log.info(
                    f"  [{scenario}] {i+1}/{len(houses)} dirs  "
                    f"houses={len(all_houses)}  pickup={len(all_pickup)}  "
                    f"recep={len(all_recep)}  trajs={n_trajs_total}  "
                    f"h5={n_h5}  elapsed={dt:.0f}s"
                )

    out = {
        "scenarios": args.scenarios,
        "src_root": str(args.src_root),
        "num_h5_files": n_h5,
        "num_trajs_scanned": n_trajs_total,
        "houses": sorted(all_houses),
        "pickup_uuids": sorted(all_pickup),
        "receptacle_uuids": sorted(all_recep),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    log.info(
        f"wrote {args.output}  houses={len(out['houses'])}  "
        f"pickup={len(out['pickup_uuids'])}  recep={len(out['receptacle_uuids'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
