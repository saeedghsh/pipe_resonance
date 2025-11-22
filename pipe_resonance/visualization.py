"""Utilities for plotting and image overlaying"""

# pylint: disable=no-member
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .configs import DrawConfig


def overlay_debug(
    *,
    frame: np.ndarray,
    marker: tuple[int | None, int | None],
    roi_x_range: tuple[int, int],
    dbg_mask: np.ndarray | None,
    draw_config: DrawConfig,
) -> np.ndarray:
    """Draw ROI, marker, and optional inset."""
    vis = frame.copy()
    cv2.rectangle(
        vis,
        (roi_x_range[0], 0),
        (roi_x_range[1], frame.shape[0]),
        draw_config.roi_color,
        draw_config.roi_thickness,
    )

    if marker[0] is not None and marker[1] is not None:
        cv2.circle(
            vis,
            (marker[0], marker[1]),
            draw_config.marker_radius,
            draw_config.marker_color,
            draw_config.marker_thickness,
        )
        cv2.line(
            vis,
            (marker[0], 0),
            (marker[0], frame.shape[0]),
            draw_config.line_color,
            draw_config.line_thickness,
        )
    if dbg_mask is not None:
        inset = cv2.resize(
            dbg_mask,
            (
                frame.shape[1] // draw_config.inset_scale,
                frame.shape[0] // draw_config.inset_scale,
            ),
        )
        inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
        h, w = inset.shape[:2]
        vis[10 : 10 + h, 10 : 10 + w] = inset
    return vis


def plot_trace(
    rows: list[tuple[int, float, int | None, int | None]], x0: float, plot_path: Path
) -> None:
    """Plot horizontal displacement."""
    ts = [t for _, t, _, _ in rows]
    xs = [np.nan if x is None else (float(x) - x0) for _, _, x, _ in rows]
    plt.figure()
    plt.plot(ts, xs)
    plt.xlabel("time [s]")
    plt.ylabel("horizontal displacement [px] (relative)")
    plt.title("Marker horizontal displacement vs time")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()
