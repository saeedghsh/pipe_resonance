"""Utilities for interactively selecting a marker and tracking it through a video"""

# pylint: disable=no-member
import csv
import math
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from .configs import DetectionConfig, TemplateConfig, TrackerConfig
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


def _match_template(  # pylint: disable=too-many-locals
    *,
    image_gray: np.ndarray,
    roi_x_range: tuple[int, int],
    y_limits: tuple[int, int],
    template: np.ndarray,
    template_config: TemplateConfig,
) -> tuple[int | None, int | None]:
    # build a search window around the current ROI, expanded by margin
    h, w = image_gray.shape
    ymin, ymax = y_limits
    sx0 = max(0, roi_x_range[0] - template_config.search_margin_px)
    sx1 = min(w, roi_x_range[1] + template_config.search_margin_px)
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


def _contour_center(contour: np.ndarray) -> tuple[int, int] | None:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)


def _contour_center_with_score(  # pylint: disable=too-many-arguments
    *,
    contour: np.ndarray,
    image_roi: np.ndarray,
    image_roi_bw: np.ndarray,
    roi_x_range: tuple[int, int],
    prev_xy: tuple[int, int] | None = None,
    search_radius: int | None = None,
) -> tuple[tuple[int, int] | None, float | None]:
    center = _contour_center(contour)
    if center is None:
        return None, None
    height, _ = image_roi_bw.shape
    mask = np.zeros_like(image_roi_bw)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_int = float(cv2.mean(image_roi, mask=mask)[0])  # type: ignore[index]
    score = mean_int - 0.01 * abs(center[1] - height / 2)
    if prev_xy is not None and search_radius is None:
        px, py = prev_xy
        px_roi = px - roi_x_range[0]
        score -= 0.2 * math.hypot(center[0] - px_roi, center[1] - py)
    return center, cast(float, score)


def _violates_radius(
    *,
    center: tuple[int, int],
    roi_x_range: tuple[int, int],
    prev_xy: tuple[int, int] | None = None,
    search_radius: int | None = None,
) -> bool:
    if prev_xy is not None and search_radius is not None:
        px, py = prev_xy
        px_roi = px - roi_x_range[0]
        if math.hypot(center[0] - px_roi, center[1] - py) > search_radius:
            return True
    return False


def _best_contour_center(  # pylint: disable=too-many-arguments
    *,
    image_roi: np.ndarray,
    image_roi_bw: np.ndarray,
    detection_config: DetectionConfig,
    roi_x_range: tuple[int, int],
    prev_xy: tuple[int, int] | None = None,
    search_radius: int | None = None,
    y_limits: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    contours, _ = cv2.findContours(image_roi_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_score: float = -1e9
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < detection_config.min_area or area > detection_config.max_area:
            continue

        center, score = _contour_center_with_score(
            contour=contour,
            image_roi=image_roi,
            image_roi_bw=image_roi_bw,
            roi_x_range=roi_x_range,
            prev_xy=prev_xy,
            search_radius=search_radius,
        )

        if center is None or score is None:
            continue
        if y_limits is not None and not (center[1] < y_limits[0] or center[1] > y_limits[1]):
            continue
        if _violates_radius(
            center=center, roi_x_range=roi_x_range, prev_xy=prev_xy, search_radius=search_radius
        ):
            continue
        if score > best_score:
            best_score = score
            best = (center[0] + roi_x_range[0], center[1])
    return best


def _enhance_contrast(image: np.ndarray, detection_config: DetectionConfig) -> np.ndarray:
    clip_limit = detection_config.clahe_clip_limit
    tile_size = detection_config.clahe_tile
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def _black_and_white(image: np.ndarray, detection_config: DetectionConfig) -> np.ndarray:
    thr_val = np.percentile(image, detection_config.thr_percentile)
    _, image_bw = cv2.threshold(image, max(1, int(thr_val)), 255, cv2.THRESH_BINARY)
    if detection_config.open_kernel > 1:
        kernel = np.ones((detection_config.open_kernel, detection_config.open_kernel), np.uint8)
        image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_OPEN, kernel, iterations=1)
    if detection_config.dilate_kernel > 1 and detection_config.dilate_iter > 0:
        kernel = np.ones((detection_config.dilate_kernel, detection_config.dilate_kernel), np.uint8)
        image_bw = cv2.dilate(image_bw, kernel, iterations=detection_config.dilate_iter)
    return image_bw


def _detect_blob(  # pylint: disable=too-many-arguments
    *,
    image_gray: np.ndarray,
    roi_x_range: tuple[int, int],
    detection_config: DetectionConfig,
    prev_xy: tuple[int, int] | None = None,
    search_radius: int | None = None,
    y_limits: tuple[int, int] | None = None,
) -> tuple[int, int, np.ndarray] | None:
    """Detect bright marker as a compact bright blob inside [roi_x0, roi_x1)."""
    image_roi = image_gray[:, roi_x_range[0] : roi_x_range[1]]
    if detection_config.use_clahe:
        image_roi = _enhance_contrast(image_roi, detection_config)
    image_roi_bw = _black_and_white(image_roi, detection_config)

    contour_center = _best_contour_center(
        image_roi_bw=image_roi_bw,
        image_roi=image_roi,
        detection_config=detection_config,
        roi_x_range=roi_x_range,
        prev_xy=prev_xy,
        search_radius=search_radius,
        y_limits=y_limits,
    )

    if contour_center is None:
        return None
    return contour_center[0], contour_center[1], image_roi_bw


def _detect_marker(  # pylint: disable=too-many-arguments
    *,
    image_gray: np.ndarray,
    roi_x_range: tuple[int, int],
    y_gate: tuple[int, int],
    first_frame: bool,
    template: np.ndarray,
    tracker_config: TrackerConfig,
    prev_xy: tuple[int, int] | None = None,
) -> tuple[int | None, int | None, np.ndarray | None]:
    x, y, dbg_mask = None, None, None
    # 1) Try template tracking first
    tx, ty = (None, None)
    if tracker_config.template.enabled and prev_xy is not None:
        tx, ty = _match_template(
            image_gray=image_gray,
            roi_x_range=roi_x_range,
            y_limits=y_gate,
            template=template,
            template_config=tracker_config.template,
        )

    if tx is not None and ty is not None:
        x, y = tx, ty
    else:
        # 2) Fallback to blob detection inside current ROI
        search_r = (
            tracker_config.gate.init_search_radius
            if first_frame
            else tracker_config.gate.track_search_radius
        )
        det = _detect_blob(
            image_gray=image_gray,
            roi_x_range=roi_x_range,
            detection_config=tracker_config.det,
            prev_xy=prev_xy,
            search_radius=search_r,
            y_limits=y_gate,
        )
        if det is not None:
            x, y, dbg_mask = det
    return x, y, dbg_mask


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


def track_video(  # pylint: disable=too-many-locals,  too-many-statements
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
    roi_x_range = (
        max(0, seed_xy[0] - cfg.roi.init_band_px),
        min(width, seed_xy[0] + cfg.roi.init_band_px),
    )
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

        x, y, dbg_mask = _detect_marker(
            image_gray=gray,
            roi_x_range=roi_x_range,
            y_gate=y_gate,
            first_frame=not frame_idx,
            template=template,
            tracker_config=cfg,
            prev_xy=prev_xy,
        )

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
            roi_x_range = (
                max(0, cx - cfg.roi.dynamic_band_px),
                min(width, cx + cfg.roi.dynamic_band_px),
            )
        else:
            # Fallback to a centered band until we recover
            band_half = int(cfg.roi.full_band_fraction * width / 2.0)
            roi_x_range = (
                max(0, width // 2 - band_half),
                min(width, width // 2 + band_half),
            )

        t = frame_idx / fps
        rows.append((frame_idx, t, x, y))

        if writer is not None:
            writer.write(
                overlay_debug(
                    frame=frame,
                    marker=(x, y),
                    roi_x_range=roi_x_range,
                    dbg_mask=dbg_mask,
                    draw_config=cfg.draw,
                )
            )

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    x0 = (
        float(np.median(xs_for_baseline))
        if xs_for_baseline
        else (roi_x_range[0] + roi_x_range[1]) / 2.0
    )
    _write_csv(csv_path, rows, x0)
    plot_trace(rows, x0, plot_path)
