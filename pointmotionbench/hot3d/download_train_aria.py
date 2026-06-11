"""
download_train_aria.py

Downloads HOT3D train_aria TAR files from the HOT3D dataset.
Each TAR is one 5-second Aria clip (~150 frames) and is the input to extract_rgbs.py.

Usage:
    # Download only clips needed for PointMotionBench (1,279 source clips):
    python hot3d/download_train_aria.py \
        --output   /path/to/train_aria \
        --captions hot3d/hot3d_annotations.json

    # Download all 1,516 train_aria clips:
    python hot3d/download_train_aria.py \
        --output /path/to/train_aria

    # Gated dataset — pass your HuggingFace token:
        --token hf_...

    # Resume: already-downloaded TARs are skipped automatically.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output',   required=True,
                        help='Directory to save train_aria TAR files')
    parser.add_argument('--captions', default=None,
                        help='hot3d_annotations.json — if given, only download clips used in PointMotionBench')
    parser.add_argument('--repo',     default='bop-benchmark/hot3d',
                        help='HuggingFace dataset repo (default: bop-benchmark/hot3d)')
    parser.add_argument('--token',    default=None,
                        help='HuggingFace access token (if the dataset is gated)')
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print('ERROR: huggingface_hub not installed. Run: pip install huggingface_hub')
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which clip IDs to download
    if args.captions:
        caps = json.load(open(args.captions))
        clip_ids = sorted({v['source_clip'] for v in caps.values()})
        print(f'Filtering to {len(clip_ids)} source clips from {args.captions}')
    else:
        print(f'Listing train_aria TARs from {args.repo} ...')
        api = HfApi(token=args.token)
        files = api.list_repo_files(args.repo, repo_type='dataset')
        tar_names = [f for f in files if f.startswith('train_aria/') and f.endswith('.tar')]
        clip_ids = sorted(Path(f).stem for f in tar_names)
        print(f'Found {len(clip_ids)} train_aria TARs')

    total = len(clip_ids)
    done = skipped = failed = 0

    for i, clip_id in enumerate(clip_ids, 1):
        dest = out_dir / f'{clip_id}.tar'
        if dest.exists():
            skipped += 1
            continue

        repo_path = f'train_aria/{clip_id}.tar'
        print(f'[{i}/{total}] {clip_id}.tar', end='  ', flush=True)
        try:
            hf_hub_download(
                repo_id=args.repo,
                repo_type='dataset',
                filename=repo_path,
                local_dir=str(out_dir),
                local_dir_use_symlinks=False,
                token=args.token,
            )
            # hf_hub_download places the file at local_dir/train_aria/clip-NNNNNN.tar
            # Move it up one level if needed
            nested = out_dir / 'train_aria' / f'{clip_id}.tar'
            if nested.exists() and not dest.exists():
                nested.rename(dest)
            print('OK')
            done += 1
        except Exception as e:
            print(f'FAILED: {e}')
            failed += 1

    print(f'\nDone: {done} downloaded, {skipped} already existed, {failed} failed')
    print(f'Output: {out_dir}')
    if failed:
        print('Re-run the script to retry failed downloads.')


if __name__ == '__main__':
    main()
