"""Entrypoint script to run the tracking"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from pipe_resonance.configs import TrackerConfig
from pipe_resonance.tracker import track_video


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track bright mid-marker with manual initialization.")
    p.add_argument("video", type=Path, help="Path to input video")
    return p.parse_args(argv)


def _derive_paths(video_path: Path) -> SimpleNamespace:
    output = video_path.parent / "output"
    output.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    return SimpleNamespace(
        csv_path=output / f"{stem}.csv",
        debug_video_path=output / f"{stem}_debug{video_path.suffix}",
        plot_path=output / f"{stem}_oscillation.png",
    )


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)

    video_path = args.video
    paths = _derive_paths(video_path)

    tracker_config = TrackerConfig()
    track_video(
        video_path=video_path,
        csv_path=paths.csv_path,
        debug_video_path=paths.debug_video_path,
        plot_path=paths.plot_path,
        cfg=tracker_config,
    )
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
