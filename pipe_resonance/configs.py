"""Configuration containers"""

from pydantic import BaseModel, Field


class ROIConfig(BaseModel):
    """Configure horizontal search bands and vertical gating."""

    full_band_fraction: float = Field(
        0.34,
        description="Fallback band as fraction of width, only used before first valid x.",
    )
    init_band_px: int = Field(60, description="Half-width around the clicked seed on frame 0.")
    dynamic_band_px: int = Field(150, description="Half-width around last x for per-frame ROI.")
    y_pad_px: int = Field(160, description="Vertical gate around seed y to exclude bottom fixture.")


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

    baseline_frames: int = Field(30, description="Frames used to compute horizontal center x0.")


class TrackerConfig(BaseModel):
    """Bundle all tracker configs."""

    roi: ROIConfig = ROIConfig()
    det: DetectionConfig = DetectionConfig()
    gate: GateConfig = GateConfig()
    out: OutputConfig = OutputConfig()
    draw: DrawConfig = DrawConfig()
    template: TemplateConfig = TemplateConfig()
