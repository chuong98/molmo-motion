"""
trim_hot3d_clips.py

Trims 150-frame source clips to PointMotionBench frame windows.

Input:
  - <src_dir>/  containing clip-NNNNNN_rgb.mp4 files
                (produced by extract_rgbs.py from train_aria TARs)
  - hot3d_annotations.json  (included in this repository)

Output:
  - <output_dir>/clip-NNNNNN_objK_s<t0>_e<t1>.mp4  per PMB clip

Usage:
    python hot3d/trim_hot3d_clips.py \
        --src_dir    /path/to/rgbs \
        --captions   hot3d/hot3d_annotations.json \
        --output_dir hot3d/videos

Dependencies: imageio[ffmpeg]  (pip install imageio[ffmpeg])
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_frames(mp4_path):
    """Read all frames of an MP4 into a list of (H, W, 3) uint8 arrays."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    reader = imageio.get_reader(str(mp4_path), 'ffmpeg')
    frames = [f for f in reader]
    reader.close()
    return frames


def write_frames(frames, out_path, fps=30):
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    writer = imageio.get_writer(
        str(out_path),
        format='FFMPEG',
        mode='I',
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p',
        macro_block_size=1,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src_dir',    required=True,
                        help='Directory containing clip-NNNNNN_rgb.mp4 source files')
    parser.add_argument('--captions',   default=str(Path(__file__).parent / 'hot3d_annotations.json'),
                        help='Path to hot3d_annotations.json (default: alongside this script)')
    parser.add_argument('--output_dir', default=str(Path(__file__).parent / 'videos'),
                        help='Output directory for trimmed video clips (default: ./videos/ alongside this script)')
    parser.add_argument('--clip_ids',   default=None,
                        help='Comma-separated PMB clip IDs to process (default: all)')
    args = parser.parse_args()

    try:
        import imageio  # noqa: F401
    except ImportError:
        print('ERROR: imageio not installed. Run: pip install imageio[ffmpeg]')
        sys.exit(1)

    src_dir = Path(args.src_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    captions = json.load(open(args.captions))

    if args.clip_ids:
        wanted = set(args.clip_ids.split(','))
        captions = {k: v for k, v in captions.items() if k in wanted}

    total = len(captions)
    done = skipped = failed = 0

    # Cache frames per source clip to avoid re-reading the same file
    frame_cache = {}  # source_clip -> list of frames

    for i, (clip_id, entry) in enumerate(sorted(captions.items()), 1):
        out_path = out_dir / f'{clip_id}.mp4'
        if out_path.exists():
            skipped += 1
            continue

        source_clip = entry['source_clip']
        t0, t1 = entry['frame_range']
        src_mp4 = src_dir / f'{source_clip}_rgb.mp4'

        if not src_mp4.exists():
            print(f'[{i}/{total}] MISSING  {source_clip}_rgb.mp4')
            failed += 1
            continue

        try:
            if source_clip not in frame_cache:
                # Evict old entries to keep memory bounded (keep last 2 sources)
                if len(frame_cache) >= 2:
                    oldest = next(iter(frame_cache))
                    del frame_cache[oldest]
                frame_cache[source_clip] = read_frames(src_mp4)

            frames = frame_cache[source_clip]
            trimmed = frames[t0:t1 + 1]
            write_frames(trimmed, out_path)
            print(f'[{i}/{total}] OK  {clip_id}  frames={t0}-{t1}  ({len(trimmed)} frames)')
            done += 1

        except Exception as e:
            print(f'[{i}/{total}] ERROR  {clip_id}: {e}')
            failed += 1

    print(f'\nDone: {done} trimmed, {skipped} already existed, {failed} failed')
    print(f'Output: {out_dir}')


if __name__ == '__main__':
    main()
