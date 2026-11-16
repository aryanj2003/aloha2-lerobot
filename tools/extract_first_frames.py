#!/usr/bin/env python3
"""Save the first frame of every camera of every ALOHA HDF5 episode, grouped by camera.

Output: <output-dir>/<camera>/<episode_name>.png  (e.g. first_frames/cam_high/episode_0.png)
Missing directories are created.
"""

import argparse
from pathlib import Path

import cv2
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path
                        , help="Folder containing episode_*.hdf5")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <dataset_dir>/first_frames")
    args = parser.parse_args()

    out = args.output_dir or args.dataset_dir / "first_frames"
    # Numeric sort so episode_10 follows episode_9.
    files = sorted(args.dataset_dir.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise SystemExit(f"No episode_*.hdf5 in {args.dataset_dir}")

    for path in files:
        try:
            with h5py.File(path, "r") as root:
                for cam, ds in root["/observations/images"].items():
                    # Stored as RGB; cv2 writes BGR.
                    (out / cam).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out / cam / f"{path.stem}.png"), cv2.cvtColor(ds[0], cv2.COLOR_RGB2BGR))
        except (OSError, KeyError) as exc:
            print(f"skip {path.name}: {exc}")
    print(f"Wrote first frames of {len(files)} episodes to {out}")


if __name__ == "__main__":
    main()
