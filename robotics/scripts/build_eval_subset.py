"""Build a deterministic, bucket-balanced subset of an upstream MolmoSpaces
benchmark for monitored per-checkpoint evaluation.

Buckets (configurable via ``--n_ss / --n_su / --n_us``):

- ``ss`` (``seen_house × seen_obj``)   — house and pickup-object both seen in train.
- ``su`` (``seen_house × unseen_obj``) — new object in a familiar scene.
- ``us`` (``unseen_house × seen_obj``) — familiar object in a new scene.

"Seen" ⇔ the entity appears in ``--train_sig`` (the JSON produced by
``scan_train_signatures.py``).

Within each bucket, selection is deterministic: episodes are grouped by
the pickup-object category (first underscore-separated token of
``pickup_obj_name``), then round-robined across categories sorted by
``(pickup_uuid, house_index)``. No RNG.

Optional ``--success_list``: a JSON file (e.g. from
``harvest_molmobot_success.py``) listing
``{(house_index, episode_idx): success}`` pairs. When set, only episodes
the listed policy can grasp successfully remain in the candidate pool —
useful for hybrid eval where you want to attribute failures to the
post-grasp phase alone.

Usage:

    python scripts/build_eval_subset.py \\
        --benchmark_dir /path/to/upstream_benchmark \\
        --train_sig train_signatures.json \\
        --dst_root /path/to/output_subset \\
        --manifest /path/to/output_subset/manifest.json \\
        --n_ss 50 --n_su 25 --n_us 25

Output:

    <dst_root>/benchmark.json           — the selected episodes (upstream schema)
    <dst_root>/benchmark_metadata.json  — copied from upstream
    <manifest>                          — auditable selection log

The output dir is consumable by the MolmoSpaces eval pipeline as a
drop-in replacement for the upstream benchmark dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


log = logging.getLogger(__name__)


def _bucket_key(ep, train_houses, train_pickup_uids):
    house = ep.get("house_index")
    obj = ep.get("task", {}).get("pickup_obj_name", "")
    uid = obj.split("_")[1] if "_" in obj else None
    hb = "seen_house" if house in train_houses else "unseen_house"
    ob = "seen_obj" if uid in train_pickup_uids else "unseen_obj"
    return hb, ob


def _pickup_category(ep) -> str:
    obj = ep.get("task", {}).get("pickup_obj_name", "")
    return obj.split("_")[0].lower() if obj else "UNKNOWN"


def _pickup_uuid(ep) -> str:
    obj = ep.get("task", {}).get("pickup_obj_name", "")
    return obj.split("_")[1] if "_" in obj else ""


def _stratified_pick(candidates: List[dict], n: int) -> List[dict]:
    """Deterministic, RNG-free selection of ``n`` episodes from
    ``candidates``: group by pickup category, then round-robin in
    sorted ``(pickup_uuid, house_index)`` order per category."""
    if n >= len(candidates):
        return sorted(
            candidates,
            key=lambda e: (_pickup_category(e), _pickup_uuid(e), e["house_index"]),
        )

    by_cat: Dict[str, List[dict]] = {}
    for ep in candidates:
        by_cat.setdefault(_pickup_category(ep), []).append(ep)
    for cat in by_cat:
        by_cat[cat] = sorted(by_cat[cat], key=lambda e: (_pickup_uuid(e), e["house_index"]))
    cat_names = sorted(by_cat.keys())
    pointers = {c: 0 for c in cat_names}
    out: List[dict] = []
    while len(out) < n:
        progressed = False
        for cat in cat_names:
            if len(out) >= n:
                break
            p = pointers[cat]
            if p < len(by_cat[cat]):
                out.append(by_cat[cat][p])
                pointers[cat] += 1
                progressed = True
        if not progressed:
            break
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_dir", type=Path, required=True)
    p.add_argument("--train_sig", type=Path, required=True)
    p.add_argument("--dst_root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--n_ss", type=int, default=50, help="seen-house × seen-obj")
    p.add_argument("--n_su", type=int, default=25, help="seen-house × unseen-obj")
    p.add_argument("--n_us", type=int, default=25, help="unseen-house × seen-obj")
    p.add_argument(
        "--success_list", type=Path, default=None,
        help="Optional JSON gating candidates to episodes the listed "
        "policy can grasp; see docstring.",
    )
    args = p.parse_args()

    sig = json.loads(args.train_sig.read_text())
    train_houses = set(sig["houses"])
    train_pickup = set(sig["pickup_uuids"])
    log.info(f"train signature: houses={len(train_houses)} pickup_uuids={len(train_pickup)}")

    bench_path = args.benchmark_dir / "benchmark.json"
    meta_path = args.benchmark_dir / "benchmark_metadata.json"
    benchmark = json.loads(bench_path.read_text())
    log.info(f"benchmark: {len(benchmark)} episodes at {bench_path}")

    # Tag every episode with its position within the upstream list + its
    # contiguous-per-house episode_idx (matches the eval runner's enumeration).
    seen_per_house: Dict[int, int] = defaultdict(int)
    for i, ep in enumerate(benchmark):
        ep.setdefault("_orig_idx", i)
        h = ep["house_index"]
        ep["_episode_idx"] = seen_per_house[h]
        seen_per_house[h] += 1

    n_before_succ = len(benchmark)
    if args.success_list is not None:
        succ = json.loads(args.success_list.read_text())
        succ_map: Dict[Tuple[int, int], bool] = {
            (int(e["house_index"]), int(e["episode_idx"])): bool(e["success"])
            for e in succ["episodes"]
        }
        before = len(benchmark)
        benchmark = [
            ep for ep in benchmark
            if succ_map.get((ep["house_index"], ep["_episode_idx"]), False)
        ]
        log.info(
            f"success-list filter: {before} → {len(benchmark)} "
            f"(dropped {before - len(benchmark)} non-successful / unevaluated)"
        )

    buckets: Dict[Tuple[str, str], List[dict]] = {
        ("seen_house", "seen_obj"): [],
        ("seen_house", "unseen_obj"): [],
        ("unseen_house", "seen_obj"): [],
        ("unseen_house", "unseen_obj"): [],
    }
    for ep in benchmark:
        buckets[_bucket_key(ep, train_houses, train_pickup)].append(ep)
    for k, eps in buckets.items():
        log.info(f"  bucket {k}: {len(eps)} candidates")

    plan = [
        (("seen_house", "seen_obj"), args.n_ss),
        (("seen_house", "unseen_obj"), args.n_su),
        (("unseen_house", "seen_obj"), args.n_us),
    ]
    for key, n_want in plan:
        if len(buckets[key]) < n_want:
            log.error(
                f"bucket {key} has only {len(buckets[key])} candidates "
                f"but {n_want} requested"
            )
            return 2

    picked: List[dict] = []
    bucket_of: Dict[int, Tuple[str, str]] = {}
    for key, n_want in plan:
        chosen = _stratified_pick(buckets[key], n_want)
        for ep in chosen:
            picked.append(ep)
            bucket_of[ep["_orig_idx"]] = key
        log.info(f"selected {len(chosen)} from bucket {key}")

    def _sort_key(ep):
        b = bucket_of[ep["_orig_idx"]]
        return (
            0 if b == ("seen_house", "seen_obj")
            else (1 if b == ("seen_house", "unseen_obj") else 2),
            ep["_orig_idx"],
        )
    picked.sort(key=_sort_key)

    args.dst_root.mkdir(parents=True, exist_ok=True)
    subset_bench = args.dst_root / "benchmark.json"
    with subset_bench.open("w") as f:
        clean = [
            {k: v for k, v in ep.items() if k not in ("_orig_idx", "_episode_idx")}
            for ep in picked
        ]
        json.dump(clean, f)
    log.info(f"wrote {subset_bench} ({len(clean)} episodes)")

    if meta_path.exists():
        shutil.copy2(meta_path, args.dst_root / "benchmark_metadata.json")
        log.info(f"copied {meta_path.name} → {args.dst_root}")

    manifest = {
        "source_benchmark": str(bench_path),
        "train_signature": str(args.train_sig),
        "success_list": str(args.success_list) if args.success_list else None,
        "n_train_houses": len(train_houses),
        "n_train_pickup_uids": len(train_pickup),
        "n_benchmark_episodes_before_filter": n_before_succ,
        "n_benchmark_episodes_after_filter": len(benchmark),
        "n_episodes": len(picked),
        "buckets": {
            "seen_house_seen_obj":   {"target": args.n_ss, "selected": 0},
            "seen_house_unseen_obj": {"target": args.n_su, "selected": 0},
            "unseen_house_seen_obj": {"target": args.n_us, "selected": 0},
        },
        "episodes": [],
    }
    for ep in picked:
        key = bucket_of[ep["_orig_idx"]]
        bname = f"{key[0]}_{key[1]}"
        manifest["buckets"][bname]["selected"] += 1
        manifest["episodes"].append({
            "orig_idx": ep["_orig_idx"],
            "episode_idx": ep["_episode_idx"],
            "house_index": ep["house_index"],
            "pickup_obj_name": ep["task"]["pickup_obj_name"],
            "pickup_category": _pickup_category(ep),
            "pickup_uuid": _pickup_uuid(ep),
            "place_receptacle_name": ep["task"]["place_receptacle_name"],
            "bucket": bname,
        })
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
