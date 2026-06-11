"""Convert a MolmoMotion FSDP-2 sharded checkpoint into the single-file
``model.pt + config.yaml`` format that the MolmoBot trainer expects to
initialize from.

A MolmoMotion training run saves each step as
``step<N>/{config.yaml, model_and_optim/}`` where ``model_and_optim/`` is
a PyTorch distcp checkpoint. The MolmoBot trainer's
``--initial_model_checkpoint`` flag expects a directory containing a single
``model.pt`` + ``config.yaml``. This script does that conversion.

Usage:

    python scripts/unshard_pretrained.py <src_ckpt> <dst_ckpt>

Both arguments are directories. ``<src_ckpt>/model_and_optim`` must exist;
``<dst_ckpt>`` will be created if missing. Run on a single rank, CPU is
sufficient (peak RAM ~17 GB for a 4B-param checkpoint; runtime ~2 min on
a typical workstation).

Requires the MolmoBot codebase on PYTHONPATH (we import ``olmo.*`` from it).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from omegaconf import OmegaConf

from olmo.models.model_config import get_model_types
from olmo.train.checkpointer import load_model_state
from olmo.util import prepare_cli_environment


log = logging.getLogger(__name__)


def unshard(src_ckpt: Path, dst_ckpt: Path) -> None:
    assert (src_ckpt / "config.yaml").exists(), f"missing config.yaml in {src_ckpt}"
    assert (src_ckpt / "model_and_optim").is_dir(), f"missing model_and_optim/ in {src_ckpt}"
    dst_ckpt.mkdir(parents=True, exist_ok=True)

    raw = OmegaConf.load(src_ckpt / "config.yaml")
    model_name = raw.model.get("model_name", "molmo")
    model_cfg_cls = get_model_types()[model_name]
    log.info(f"model_name={model_name} -> {model_cfg_cls.__name__}")

    # Schema-validated load of just the model section. Legacy-config hooks
    # absorb fields specific to the MolmoMotion training schema.
    model_cfg = model_cfg_cls.load(src_ckpt / "config.yaml", key="model")

    log.info("building model on meta, materializing on cpu")
    with torch.device("meta"):
        model = model_cfg.build_model()
    model.to_empty(device=torch.device("cpu"))

    log.info(f"loading distcp shards from {src_ckpt}/model_and_optim")
    load_model_state(src_ckpt, model)

    log.info("gathering full state_dict and saving to model.pt")
    sd = dist_cp_sd.get_model_state_dict(
        model,
        options=dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    torch.save(sd, dst_ckpt / "model.pt")
    n_gib = (dst_ckpt / "model.pt").stat().st_size / 2 ** 30
    log.info(f"  wrote {n_gib:.2f} GiB -> {dst_ckpt / 'model.pt'}")

    # Model-only config.yaml — MolmoBot's loader expects this shape.
    out_cfg = {
        "run_name": raw.get("run_name", "unsharded"),
        "model": OmegaConf.to_container(raw.model, resolve=True),
    }
    OmegaConf.save(OmegaConf.create(out_cfg), dst_ckpt / "config.yaml")
    log.info(f"  wrote {dst_ckpt / 'config.yaml'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "src_ckpt",
        type=Path,
        help="source MolmoMotion checkpoint dir (containing config.yaml + model_and_optim/)",
    )
    parser.add_argument(
        "dst_ckpt",
        type=Path,
        help="destination dir for model.pt + config.yaml (will be created)",
    )
    args = parser.parse_args()

    prepare_cli_environment()
    unshard(args.src_ckpt, args.dst_ckpt)


if __name__ == "__main__":
    main()
