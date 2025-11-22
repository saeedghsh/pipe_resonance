"""Entrypoint script to run the tracking"""

import argparse
import os
import sys

from pipe_resonance.configs import TrackerConfig
from pipe_resonance.tracker import track_video


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track bright mid-marker with manual initialization.")
    p.add_argument("video", help="Path to input video")
    p.add_argument("--out-csv", default="marker_trace.csv", help="Output CSV path")
    p.add_argument("--debug-video", default=None, help="Optional MP4 with overlay")
    p.add_argument("--plot", action="store_true", help="Show a quick matplotlib plot")
    return p.parse_args(argv)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    cfg = TrackerConfig()
    track_video(
        args.video,
        out_csv=args.out_csv,
        debug_video=args.debug_video,
        show_plot=args.plot,
        cfg=cfg,
    )
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
