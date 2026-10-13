"""Run settings.

Every tunable parameter lives here, with its default. A YAML file (see
``config/default.yaml``) can override any subset of them. The resolved settings
are written into each run's manifest, so every output is traceable to the exact
parameters that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GridConfig:
    base_res_m: float = 30.0      # terrain / lineament grid (SRTM native ~30 m)
    emit_factor: int = 2          # EMIT grid = 60 m (native EMIT pixel)
    aster_factor: int = 3         # ASTER GED grid = 90 m (closest multiple to native 100 m)
    align_m: float = 180.0        # origin snapped to this so all grids share edges
    structure_pad_m: float = 1500.0  # extra DEM around the AOI so edge lineaments are complete


@dataclass
class AcquireConfig:
    ee_project: str | None = None
    start_date: str = "2022-08-01"
    end_date: str | None = None         # None = today
    min_aoi_coverage: float = 0.05      # skip EMIT scenes covering less of the AOI than this
    max_request_mb: float = 30.0        # Earth Engine computePixels payload limit is ~48 MB
    retries: int = 4


@dataclass
class SpectralConfig:
    mc_draws: int = 64                  # Monte-Carlo draws for uncertainty propagation
    seed: int = 42
    shoulder_halfwidth_nm: float = 10.0  # shoulders are averaged over +/- this window
    detect_sigma: float = 2.0           # detection = depth > detect_sigma * sigma
    min_depth: float = 0.01             # ... and depth above this absolute floor
    ndvi_max: float = 0.30              # pixels greener than this are masked as vegetation
    min_albedo: float = 0.04            # darker pixels (shadow, water, burn scars) are masked
    chunk_pixels: int = 4096


@dataclass
class L2BConfig:
    group2_min_fit: float = 0.30        # 2 um vibrational group (Tetracorder group 2)
    group2_max_unc: float = 0.025
    group1_min_fit: float = 0.25        # 1 um electronic Fe group (Tetracorder group 1)
    group1_max_unc: float = 0.020


@dataclass
class StructureConfig:
    n_azimuths: int = 8                 # hillshade illumination directions (every 360/n deg)
    sun_elevation_deg: float = 30.0
    smooth_sigma_px: float = 1.0
    canny_low_quantile: float = 0.80    # adaptive Canny thresholds (terrain independent)
    canny_high_quantile: float = 0.95
    hough_threshold: int = 40
    min_segment_m: float = 450.0
    max_gap_m: float = 90.0
    merge_angle_deg: float = 12.0
    merge_perp_m: float = 90.0
    merge_gap_m: float = 300.0
    dedup_angle_deg: float = 10.0
    dedup_perp_m: float = 60.0
    min_length_m: float = 800.0
    density_radius_m: float = 1000.0
    support_min_fraction: float = 0.5   # share of a line that must lie on edges of one azimuth
    local_contrast: bool = True         # CLAHE on hillshades + local normalisation of ridge filter
    clahe_window_m: float = 1500.0
    clahe_clip: float = 0.01
    ridge_valley: bool = True           # add Hessian ridge/valley axes to the edge maps
    ridge_sigmas_m: tuple = (30.0, 60.0)
    ridge_quantile: float = 0.90
    profile_offset_m: float = 90.0      # cross-profile sample distance either side of a line
    scarp_min_step_m: float = 2.0       # one-sided relief needed to call a line a scarp
    scarp_step_ratio: float = 1.0       # scarp if step > ratio * |centre - mean(sides)|
    exclude_scarps: bool = True         # leave scarps out of distance / density / S_prox
    dem_source: str = "srtm"            # srtm | aw3d30 | copernicus
    canny_tile_m: float = 0.0           # > 0: Canny quantile thresholds per tile of this size (0 = whole grid)


@dataclass
class AlterationConfig:
    """Four alteration layers (methodology sections 3-4)."""
    sigma_core_m: float = 90.0          # core vent field kernel (4.3)
    sigma_halo_m: float = 375.0         # hydrothermal halo kernel (4.3)
    theta_decay: float = 0.05           # confidence = 1 - exp(-W_halo / theta)
    reference_percentile: float = 98.0  # a = clip(F / P98(F > 0), 0, 1)   (4.4)
    sericite_position_nm: tuple = (2190.0, 2215.0)   # Al-OH minimum accepted as white mica
    propylitic_l2b_classes: tuple = ("chlorite", "epidote", "prehnite", "amphibole")
    quartz_weights: tuple = (0.55, 0.45)             # QSI = 0.55 QI + 0.45 SI   (4.5)
    quartz_background_percentile: float = 15.0      # a_qtz = (QSI - P15) / (P98 - P15)
    # EMIT L2B evidence in the alteration layers (weight = fit / uncertainty, as in the legacy method):
    #   "none"           L2A detections only (default)
    #   "combine_all"    add QC-passed L2B identifications everywhere (incl. vegetation-masked pixels)
    #   "combine_valid"  add L2B only on pixels where L2A has a valid observation
    l2b_evidence: str = "none"
    l2b_classes: dict = field(default_factory=lambda: {
        "muscovite": ["white_mica"], "chlorite": ["chlorite", "epidote"], "gossan": ["ferric_oxide", "jarosite"]})
    # Lithology background removal: subtract the per-unit median of each layer (mapped rock units),
    # keep the positive residual ("stronger than normal for this rock"), then P98-normalise.
    lithology_background: bool = False
    lithology_file: str | None = None      # default: data/<aoi>/ground_truth/lithology*.geojson
    lithology_field: str = "LITHOLOGICAL_UNIT"
    min_unit_area_km2: float = 0.25        # smaller units (and unmapped ground) use the block median
    min_support: float = 0.20           # usable-pixel fraction in the halo below which a map shows "no data"
    support_sigma_m: float | None = None  # neighbourhood for data support: None = sigma_halo / 2 (~190 m),
                                          # > 0 = Gaussian of this sigma, 0 = pixel or an immediate neighbour usable
    display_upsample: int = 4


@dataclass
class TargetConfig:
    """Copper Prospectivity Index and drill-target selection (methodology sections 5-6)."""
    w_hydrothermal: float = 0.70
    w_structure: float = 0.30
    w_muscovite: float = 0.30
    w_quartz: float = 0.28
    w_gossan: float = 0.22
    w_chlorite: float = 0.20
    structure_lambda_m: float = 250.0   # S_prox = exp(-d / lambda)   (4.6)
    min_cpi: float = 0.25
    n_targets: int = 50
    min_spacing_m: float = 250.0
    tier1: float = 0.70
    tier2: float = 0.50
    tier3: float = 0.28
    require_data_support: bool = True   # no field-check locations where EMIT had no usable pixels
    zone_min_area_km2: float = 0.01     # prospective zones smaller than this (~11 cells) are dropped


@dataclass
class Settings:
    grid: GridConfig = field(default_factory=GridConfig)
    acquire: AcquireConfig = field(default_factory=AcquireConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    l2b: L2BConfig = field(default_factory=L2BConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    alteration: AlterationConfig = field(default_factory=AlterationConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    run_label: str = ""   # shown on every map, e.g. "EXTENDED COVERAGE (NDVI <= 0.35) - lower confidence"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply(obj: Any, values: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in (values or {}).items():
        if key not in known:
            raise KeyError(f"Unknown setting '{path}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    settings = Settings()
    if path:
        with open(path, encoding="utf-8") as f:
            _apply(settings, yaml.safe_load(f) or {})
    if overrides:
        _apply(settings, overrides)
    return settings
