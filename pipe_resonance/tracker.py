"""Utilities for interactively selecting a marker and tracking it through a video"""

# pylint: disable=no-member
import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .configs import TemplateConfig, TrackerConfig
from .visualization import overlay_debug, plot_trace


def _extract_patch(gray: np.ndarray, center_xy: tuple[int, int], half: int) -> np.ndarray:
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
    *,
    image_gray: np.ndarray,
    roi_x0: int,
    roi_x1: int,
    y_limits: tuple[int, int],
    template: np.ndarray,
    template_config: TemplateConfig,
) -> tuple[int | None, int | None]:
    # build a search window around the current ROI, expanded by margin
    h, w = image_gray.shape
    ymin, ymax = y_limits
    sx0 = max(0, roi_x0 - template_config.search_margin_px)
    sx1 = min(w, roi_x1 + template_config.search_margin_px)
    sy0 = max(0, ymin)
    sy1 = min(h, ymax)
    search = image_gray[sy0:sy1, sx0:sx1]
    if not search.size:
        return None, None

    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    __min_val, max_val, __min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val < template_config.min_corr:
        return None, None

    th, tw = template.shape
    cx = sx0 + max_loc[0] + tw // 2
    cy = sy0 + max_loc[1] + th // 2
    return cx, cy


def detect_marker(
    gray: np.ndarray,
    roi_x0: int,
    roi_x1: int,
    cfg: TrackerConfig,
    prev_xy: tuple[int, int] | None = None,
    search_radius: int | None = None,
    y_limits: tuple[int, int] | None = None,
) -> tuple[int, int, np.ndarray] | None:
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
        mean_int = float(cv2.mean(roi_eq, mask=mask)[0])  # type: ignore[index]

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


def _get_manual_seed(cap: cv2.VideoCapture) -> tuple[int, int]:
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame for manual init.")

    scale = 2.5  # about 2–3x larger on screen
    disp_w = int(frame0.shape[1] * scale)
    disp_h = int(frame0.shape[0] * scale)

    seed: tuple[int, int] | None = None

    def on_mouse(event: int, x: int, y: int, __flags: int, __param: Any) -> None:
        nonlocal seed
        if event == cv2.EVENT_LBUTTONDOWN:
            # map click from display space back to original frame coords
            seed = (int(x / scale), int(y / scale))

    win = "click marker, then press space"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = cv2.resize(frame0, (disp_w, disp_h))
        if seed is not None:
            cx, cy = int(seed[0] * scale), int(seed[1] * scale)
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
            if seed is not None:
                break
        elif k in (ord("r"), ord("R")):
            seed = None
        elif k in (27, ord("q"), ord("Q")):
            raise SystemExit("User aborted manual init.")

    cv2.destroyWindow(win)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if seed is not None:
        return seed

    raise RuntimeError("seed remained None")


def _write_csv(
    csv_path: Path, rows: list[tuple[int, float, int | None, int | None]], x0: float
) -> None:
    """Write frame, time, x, y, x_rel to CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "x", "y", "x_rel"])
        for f_idx, t, x, y in rows:
            x_rel = None if x is None else (float(x) - x0)
            w.writerow(
                [
                    f_idx,
                    f"{t:.6f}",
                    "" if x is None else x,
                    "" if y is None else y,
                    "" if x_rel is None else f"{x_rel:.3f}",
                ]
            )


def _make_writer(path: str | None, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
    """Create video writer if path is provided."""
    if not path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def track_video(
    *, video_path: Path, csv_path: Path, debug_video_path: Path, plot_path: Path, cfg: TrackerConfig
) -> None:
    """Track marker with manual seed, dynamic ROI, and gated search."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Manual seed (bigger window already implemented in your _get_manual_seed)
    seed_xy = _get_manual_seed(cap)

    # Persistent vertical gate around seed to avoid the bottom fixture
    y_gate = (
        max(0, seed_xy[1] - cfg.roi.y_pad_px),
        min(height - 1, seed_xy[1] + cfg.roi.y_pad_px),
    )

    # Initial horizontal band around the click
    roi_x0 = max(0, seed_xy[0] - cfg.roi.init_band_px)
    roi_x1 = min(width, seed_xy[0] + cfg.roi.init_band_px)

    writer = _make_writer(str(debug_video_path), fps, width, height)

    rows: list[tuple[int, float, int | None, int | None]] = []
    xs_for_baseline: list[int] = []

    prev_xy: tuple[int, int] | None = seed_xy
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
        tx, ty = (None, None)
        if cfg.template.enabled and prev_xy is not None:
            tx, ty = _match_template(
                image_gray=gray,
                roi_x0=roi_x0,
                roi_x1=roi_x1,
                y_limits=y_gate,
                template=template,
                template_config=cfg.template,
            )

        if tx is not None and ty is not None:
            x, y = tx, ty
        else:
            # 2) Fallback to blob detection inside current ROI
            search_r = (
                cfg.gate.init_search_radius if not frame_idx else cfg.gate.track_search_radius
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
                    (1.0 - cfg.template.update_alpha) * old + cfg.template.update_alpha * new_patch
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

    x0 = float(np.median(xs_for_baseline)) if xs_for_baseline else (roi_x0 + roi_x1) / 2.0
    _write_csv(csv_path, rows, x0)
    plot_trace(rows, x0, plot_path)
