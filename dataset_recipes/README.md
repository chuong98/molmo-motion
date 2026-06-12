# MolmoMotion-1M — dataset construction

The MolmoMotion-1M training corpus — including every per-dataset
reconstruction recipe — lives on HuggingFace:

**[`allenai/molmo-motion-1m`](https://huggingface.co/datasets/allenai/molmo-motion-1m)**

For each of the seven sub-datasets (egodex, ytvis, hdepic, xperience,
stereo4d, droid, molmospaces) the dataset repo ships:

- `README.md` — **authoritative** schema + reconstruction procedure for
  that dataset
- `annotations/` — clips / split / index JSONs (the source of truth)
- `tracks/` — the 3D (and, for most datasets, 2D) point tracks we produced
  (`track_index/` for stereo4d, which reconstructs its tracks locally)
- `camera/` — per-frame pose + intrinsics (absent for xperience, which
  reconstructs it)
- `videos/` — shipped only for molmospaces; every other dataset
  reconstructs videos locally from the original upstream source via the
  bundled `reconstruct_*` script, due to license restrictions

Download:

```bash
huggingface-cli download allenai/molmo-motion-1m \
    --repo-type dataset --local-dir $MOLMO_MOTION_1M_ROOT
```

Then follow each `<dataset>/README.md` inside the downloaded root to
reconstruct the videos (and the per-dataset remaining signals). Nothing in
this directory is needed for training — point `MOLMO_MOTION_1M_ROOT` at
the downloaded corpus and go.
