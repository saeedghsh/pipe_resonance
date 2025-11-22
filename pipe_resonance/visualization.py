"""Utilities for plotting and image overlaying"""

# pylint: disable=no-member
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .configs import TrackerConfig


def overlay_debug(
    frame: np.ndarray,
    x: int | None,
    y: int | None,
    roi_x0: int,
    roi_x1: int,
    dbg_mask: np.ndarray | None,
    cfg: TrackerConfig,
) -> np.ndarray:
    """Draw ROI, marker, and optional inset."""
    vis = frame.copy()
    cv2.rectangle(
        vis,
        (roi_x0, 0),
        (roi_x1, frame.shape[0]),
        cfg.draw.roi_color,
        cfg.draw.roi_thickness,
    )
    if x is not None and y is not None:
        cv2.circle(
            vis,
            (x, y),
            cfg.draw.marker_radius,
            cfg.draw.marker_color,
            cfg.draw.marker_thickness,
        )
        cv2.line(
            vis,
            (x, 0),
            (x, frame.shape[0]),
            cfg.draw.line_color,
            cfg.draw.line_thickness,
        )
    if dbg_mask is not None:
        inset = cv2.resize(
            dbg_mask,
            (
                frame.shape[1] // cfg.draw.inset_scale,
                frame.shape[0] // cfg.draw.inset_scale,
            ),
        )
        inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
        h, w = inset.shape[:2]
        vis[10 : 10 + h, 10 : 10 + w] = inset
    return vis


def plot_trace(rows: list[tuple[int, float, int | None, int | None]], x0: float) -> None:
    """Plot horizontal displacement."""
    ts = [t for _, t, _, _ in rows]
    xs = [np.nan if x is None else (float(x) - x0) for _, _, x, _ in rows]
    plt.figure()
    plt.plot(ts, xs)
    plt.xlabel("time [s]")
    plt.ylabel("horizontal displacement [px] (relative)")
    plt.title("Marker horizontal displacement vs time")
    plt.grid(True)
    plt.savefig("oscillation.png")
    plt.show()
