#!/usr/bin/env python

# "air_set/WhatsApp Video 2025-04-19 at 20.04.04.mp4"
# "air_set/WhatsApp Video 2025-04-19 at 20.06.10.mp4"

# python main.py "air_set/WhatsApp Video 2025-04-19 at 20.04.04.mp4" --manual-init --debug-video debug.mp4 --plot
# ffmpeg -i debug.mp4 -vcodec libx264 -pix_fmt yuv420p -preset veryfast -crf 23 -movflags +faststart debug_wapp_comatible.mp4

import argparse
import csv
import math
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field


class ROIConfig(BaseModel):
    """Configure horizontal search bands and vertical gating."""

    full_band_fraction: float = Field(
        0.34,
        description="Fallback band as fraction of width, only used before first valid x.",
    )
    init_band_px: int = Field(
        60, description="Half-width around the clicked seed on frame 0."
    )
    dynamic_band_px: int = Field(
        150, description="Half-width around last x for per-frame ROI."
    )
    y_pad_px: int = Field(
        160, description="Vertical gate around seed y to exclude bottom fixture."
    )


class DetectionConfig(BaseModel):
    """Configure blob detection and pre-processing."""

    thr_percentile: float = Field(
        97.0,
        description="High percentile for adaptive threshold; lower if under-detecting.",
    )
    min_area: int = Field(4, description="Reject tiny specks.")
    max_area: int = Field(900, description="Reject big blobs (e.g., fixtures).")
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile: int = 8
    open_kernel: int = 3
    dilate_kernel: int = 3
    dilate_iter: int = 2


class GateConfig(BaseModel):
    """Configure temporal and spatial gates."""

    init_search_radius: int = Field(80, description="Radius around seed on frame 0.")
    track_search_radius: int = Field(90, description="Radius around last detection.")


class DrawConfig(BaseModel):
    """Configure visualization overlays."""

    roi_color: tuple[int, int, int] = (80, 80, 80)
    roi_thickness: int = 1
    marker_radius: int = 6
    marker_color: tuple[int, int, int] = (0, 255, 0)
    marker_thickness: int = 2
    line_color: tuple[int, int, int] = (0, 255, 0)
    line_thickness: int = 1
    inset_scale: int = 6  # downscale factor for dbg mask inset


class TemplateConfig(BaseModel):
    enabled: bool = True
    patch_half: int = 12  # template size = 2*half+1
    min_corr: float = 0.45  # accept match if corr >= this
    update_alpha: float = 0.15  # EMA update of template
    search_margin_px: int = 160  # extra margin around current ROI when matching


class OutputConfig(BaseModel):
    """Configure outputs and post-processing."""

    baseline_frames: int = Field(
        30, description="Frames used to compute horizontal center x0."
    )


class TrackerConfig(BaseModel):
    """Bundle all tracker configs."""

    roi: ROIConfig = ROIConfig()
    det: DetectionConfig = DetectionConfig()
    gate: GateConfig = GateConfig()
    draw: DrawConfig = DrawConfig()
    out: OutputConfig = OutputConfig()
    template: TemplateConfig = TemplateConfig()


def _extract_patch(
    gray: np.ndarray, center_xy: Tuple[int, int], half: int
) -> np.ndarray:
    x, y = center_xy
    x0 = max(0, x - half)
    y0 = max(0, y - half)
    x1 = min(gray.shape[1], x + half + 1)
    y1 = min(gray.shape[0], y + half + 1)
    patch = gray[y0:y1, x0:x1]
    # ensure fixed size by padding if at border
    h, w = patch.shape[:2]
    if h != 2 * half + 1 or w != 2 * half + 1:
        padded = np.zeros((2 * half + 1, 2 * half + 1), dtype=gray.dtype)
        padded[:h, :w] = patch
        patch = padded
    return patch


def _match_template(
    gray: np.ndarray,
    roi_x0: int,
    roi_x1: int,
    y_limits: Tuple[int, int],
    prev_xy: Tuple[int, int],
    template: np.ndarray,
    cfg: TemplateConfig,
) -> Tuple[Optional[int], Optional[int], float]:
    # build a search window around the current ROI, expanded by margin
    h, w = gray.shape
    ymin, ymax = y_limits
    sx0 = max(0, roi_x0 - cfg.search_margin_px)
    sx1 = min(w, roi_x1 + cfg.search_margin_px)
    sy0 = max(0, ymin)
    sy1 = min(h, ymax)
    search = gray[sy0:sy1, sx0:sx1]
    if search.size == 0:
        return None, None, -1.0

    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    if maxVal < cfg.min_corr:
        return None, None, maxVal

    th, tw = template.shape
    cx = sx0 + maxLoc[0] + tw // 2
    cy = sy0 + maxLoc[1] + th // 2
    return cx, cy, maxVal


def detect_marker(
    gray: np.ndarray,
    roi_x0: int,
    roi_x1: int,
    cfg: TrackerConfig,
    prev_xy: Optional[Tuple[int, int]] = None,
    search_radius: Optional[int] = None,
    y_limits: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int, np.ndarray]]:
    """Detect bright marker as a compact bright blob inside [roi_x0, roi_x1)."""
    roi = gray[:, roi_x0:roi_x1]

    if cfg.det.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.det.clahe_clip_limit,
            tileGridSize=(cfg.det.clahe_tile, cfg.det.clahe_tile),
        )
        roi_eq = clahe.apply(roi)
    else:
        roi_eq = roi

    thr_val = np.percentile(roi_eq, cfg.det.thr_percentile)
    _, bw = cv2.threshold(roi_eq, max(1, int(thr_val)), 255, cv2.THRESH_BINARY)

    if cfg.det.open_kernel > 1:
        bw = cv2.morphologyEx(
            bw,
            cv2.MORPH_OPEN,
            np.ones((cfg.det.open_kernel, cfg.det.open_kernel), np.uint8),
            iterations=1,
        )
    if cfg.det.dilate_kernel > 1 and cfg.det.dilate_iter > 0:
        bw = cv2.dilate(
            bw,
            np.ones((cfg.det.dilate_kernel, cfg.det.dilate_kernel), np.uint8),
            iterations=cfg.det.dilate_iter,
        )

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1e9
    h, _ = roi.shape

    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.det.min_area or area > cfg.det.max_area:
            continue

        m = cv2.moments(c)
        if abs(m["m00"]) < 1e-6:
            continue

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        if y_limits is not None:
            ymin, ymax = y_limits
            if cy < ymin or cy > ymax:
                continue

        if prev_xy is not None and search_radius is not None:
            px, py = prev_xy
            px_roi = px - roi_x0
            if math.hypot(cx - px_roi, cy - py) > search_radius:
                continue

        mask = np.zeros_like(bw)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_int = float(cv2.mean(roi_eq, mask=mask)[0])

        score = mean_int - 0.01 * abs(cy - h / 2)
        if prev_xy is not None and search_radius is None:
            px, py = prev_xy
            px_roi = px - roi_x0
            score -= 0.2 * math.hypot(cx - px_roi, cy - py)

        if score > best_score:
            best_score = score
            best = (cx + roi_x0, cy)

    if best is None:
        return None
    return best[0], best[1], bw


def make_writer(
    path: Optional[str], fps: float, width: int, height: int
) -> Optional[cv2.VideoWriter]:
    """Create video writer if path is provided."""
    if not path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def overlay_debug(
    frame: np.ndarray,
    x: Optional[int],
    y: Optional[int],
    roi_x0: int,
    roi_x1: int,
    dbg_mask: Optional[np.ndarray],
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


def write_csv(
    out_csv: str, rows: List[Tuple[int, float, Optional[int], Optional[int]]], x0: float
) -> None:
    """Write frame, time, x, y, x_rel to CSV."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "x", "y", "x_rel"])
        for fidx, t, x, y in rows:
            x_rel = None if x is None else (float(x) - x0)
            w.writerow(
                [
                    fidx,
                    f"{t:.6f}",
                    "" if x is None else x,
                    "" if y is None else y,
                    "" if x_rel is None else f"{x_rel:.3f}",
                ]
            )


def plot_trace(
    rows: List[Tuple[int, float, Optional[int], Optional[int]]], x0: float
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
    plt.savefig("oscillation.png")
    plt.show()


def get_manual_seed(cap: cv2.VideoCapture):
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame for manual init.")

    scale = 2.5  # about 2–3x larger on screen
    disp_w = int(frame0.shape[1] * scale)
    disp_h = int(frame0.shape[0] * scale)

    seed = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # map click from display space back to original frame coords
            seed["pt"] = (int(x / scale), int(y / scale))

    win = "click marker, then press space"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = cv2.resize(frame0, (disp_w, disp_h))
        if seed["pt"] is not None:
            cx, cy = int(seed["pt"][0] * scale), int(seed["pt"][1] * scale)
            cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 2)
        cv2.putText(
            vis,
            "Left-click near marker; space=confirm, r=reset, q=quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32):  # enter or space
            if seed["pt"] is not None:
                break
        elif k in (ord("r"), ord("R")):
            seed["pt"] = None
        elif k in (27, ord("q"), ord("Q")):
            raise SystemExit("User aborted manual init.")

    cv2.destroyWindow(win)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame0, seed["pt"]


def track_video(
    video_path: str,
    out_csv: str,
    debug_video: Optional[str],
    show_plot: bool,
    cfg: TrackerConfig,
) -> None:
    """Track marker with manual seed, dynamic ROI, and gated search."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Manual seed (bigger window already implemented in your get_manual_seed)
    _, seed_xy = get_manual_seed(cap)

    # Persistent vertical gate around seed to avoid the bottom fixture
    y_gate = (
        max(0, seed_xy[1] - cfg.roi.y_pad_px),
        min(height - 1, seed_xy[1] + cfg.roi.y_pad_px),
    )

    # Initial horizontal band around the click
    roi_x0 = max(0, seed_xy[0] - cfg.roi.init_band_px)
    roi_x1 = min(width, seed_xy[0] + cfg.roi.init_band_px)

    writer = make_writer(debug_video, fps, width, height)

    rows: List[Tuple[int, float, Optional[int], Optional[int]]] = []
    xs_for_baseline: List[int] = []

    prev_xy: Optional[Tuple[int, int]] = seed_xy
    frame_idx = 0

    # Build initial template from the seed
    tpl_half = cfg.template.patch_half
    # get first frame again to build template on the same view we track
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first frame to build template.")
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    template = _extract_patch(first_gray, seed_xy, tpl_half)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, dbg_mask = None, None, None

        # 1) Try template tracking first
        tx, ty, corr = (None, None, -1.0)
        if cfg.template.enabled and prev_xy is not None:
            tx, ty, corr = _match_template(
                gray, roi_x0, roi_x1, y_gate, prev_xy, template, cfg.template
            )

        if tx is not None and ty is not None:
            x, y = tx, ty
        else:
            # 2) Fallback to blob detection inside current ROI
            search_r = (
                cfg.gate.init_search_radius
                if frame_idx == 0
                else cfg.gate.track_search_radius
            )
            det = detect_marker(
                gray,
                roi_x0,
                roi_x1,
                cfg,
                prev_xy=prev_xy,
                search_radius=search_r,
                y_limits=y_gate,
            )
            if det is not None:
                x, y, dbg_mask = det

        # 3) If we have a position, update state and refresh template a little
        if x is not None and y is not None:
            prev_xy = (x, y)
            if len(xs_for_baseline) < cfg.out.baseline_frames:
                xs_for_baseline.append(x)
            if cfg.template.enabled:
                new_patch = _extract_patch(gray, (x, y), tpl_half).astype(np.float32)
                old = template.astype(np.float32)
                template = (
                    (1.0 - cfg.template.update_alpha) * old
                    + cfg.template.update_alpha * new_patch
                ).astype(template.dtype)

        # Update the ROI horizontally around the most recent x (or keep last)
        if prev_xy is not None:
            cx = prev_xy[0]
            roi_x0 = max(0, cx - cfg.roi.dynamic_band_px)
            roi_x1 = min(width, cx + cfg.roi.dynamic_band_px)
        else:
            # Fallback to a centered band until we recover
            band_half = int(cfg.roi.full_band_fraction * width / 2.0)
            roi_x0 = max(0, width // 2 - band_half)
            roi_x1 = min(width, width // 2 + band_half)

        t = frame_idx / fps
        rows.append((frame_idx, t, x, y))

        if writer is not None:
            writer.write(overlay_debug(frame, x, y, roi_x0, roi_x1, dbg_mask, cfg))

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    x0 = (
        float(np.median(xs_for_baseline))
        if xs_for_baseline
        else (roi_x0 + roi_x1) / 2.0
    )
    write_csv(out_csv, rows, x0)

    if show_plot:
        plot_trace(rows, x0)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Track bright mid-marker with manual initialization."
    )
    p.add_argument("video", help="Path to input video")
    p.add_argument("--out-csv", default="marker_trace.csv", help="Output CSV path")
    p.add_argument("--debug-video", default=None, help="Optional MP4 with overlay")
    p.add_argument("--plot", action="store_true", help="Show a quick matplotlib plot")
    args = p.parse_args()

    cfg = TrackerConfig()  # defaults tuned for your footage
    track_video(
        args.video,
        out_csv=args.out_csv,
        debug_video=args.debug_video,
        show_plot=args.plot,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
