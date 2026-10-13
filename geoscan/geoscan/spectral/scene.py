"""Per-scene EMIT L2A processing: quality classes, vegetation indices, absorption features."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import SpectralConfig
from ..raster_io import read_raster, write_raster
from .band_depth import FeatureMeasure, window_mean
from .features import ALBEDO_NM, CAI_NM, FEATURES, NIR_NM, RED_NM

log = logging.getLogger(__name__)

# Pixel quality classes (lowest applicable code wins)
QA_VALID, QA_OUTSIDE, QA_NODATA, QA_FLAGGED, QA_DARK, QA_VEGETATION = 0, 1, 2, 3, 4, 5
QA_LABELS = {QA_VALID: "valid", QA_OUTSIDE: "outside AOI", QA_NODATA: "no data / outside swath",
             QA_FLAGGED: "cloud / cirrus / water / spacecraft (EMIT aggregate flag)",
             QA_DARK: "dark (shadow, water, burn)", QA_VEGETATION: "green vegetation (NDVI)"}


@dataclass
class SceneResult:
    scene_id: str
    datetime: str
    layers: dict[str, np.ndarray]   # 2-D arrays on the EMIT grid
    stats: dict


def process_scene(scene_dir: Path, inside_aoi: np.ndarray, cfg: SpectralConfig, out_path: Path) -> SceneResult:
    # All-NaN pixels (outside swath) legitimately produce empty-slice warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _process_scene(scene_dir, inside_aoi, cfg, out_path)


def _process_scene(scene_dir: Path, inside_aoi: np.ndarray, cfg: SpectralConfig, out_path: Path) -> SceneResult:
    meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
    wl = np.asarray(meta["wavelengths_nm"])
    refl, grid, _, _ = read_raster(scene_dir / "reflectance.tif")
    unc, _, _, _ = read_raster(scene_dir / "uncertainty.tif")
    qa_raw, _, qa_names, _ = read_raster(scene_dir / "qa.tif")
    if inside_aoi.shape != grid.shape:
        raise ValueError("AOI mask does not match EMIT grid")

    cube = np.moveaxis(refl, 0, -1)          # (rows, cols, bands)
    ucube = np.moveaxis(unc, 0, -1)
    # EMIT sets bands in the deep water-vapour windows to -0.01; treat any negative value as missing.
    cube = np.where(cube < 0, np.nan, cube)

    ndvi_red, ndvi_nir = window_mean(cube, wl, *RED_NM), window_mean(cube, wl, *NIR_NM)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (ndvi_nir - ndvi_red) / (ndvi_nir + ndvi_red)
    cai = 100.0 * (0.5 * (window_mean(cube, wl, *CAI_NM["r2030"]) + window_mean(cube, wl, *CAI_NM["r2210"]))
                   - window_mean(cube, wl, *CAI_NM["r2100"]))
    albedo = np.nanmean(np.concatenate([cube[..., (wl >= lo) & (wl <= hi)] for lo, hi in ALBEDO_NM], axis=-1),
                        axis=-1)

    flagged = qa_raw[qa_names.index("aggregate_flag")] > 0.5
    nodata = ~np.isfinite(albedo)

    qa = np.full(grid.shape, QA_VALID, dtype="uint8")
    qa[ndvi > cfg.ndvi_max] = QA_VEGETATION
    qa[albedo < cfg.min_albedo] = QA_DARK
    qa[flagged] = QA_FLAGGED
    qa[nodata] = QA_NODATA
    qa[~inside_aoi] = QA_OUTSIDE

    layers: dict[str, np.ndarray] = {"qa": qa.astype("float32"), "ndvi": ndvi, "cai": cai, "albedo": albedo}

    # Measure every pixel with data (not only valid ones) so masked areas can still be inspected.
    measurable = ~nodata & ~flagged
    rows, cols = np.nonzero(measurable)
    rng = np.random.default_rng(cfg.seed)
    for feat in FEATURES:
        fm = FeatureMeasure(feat, wl, cfg.shoulder_halfwidth_nm)
        res = fm.measure(cube[rows, cols], ucube[rows, cols], cfg.mc_draws, rng, cfg.chunk_pixels)
        for k, v in res.items():
            arr = np.full(grid.shape, np.nan, dtype="float32")
            arr[rows, cols] = v
            layers[f"{feat.key}_{k}"] = arr

    names = list(layers)
    write_raster(out_path, np.stack([layers[n] for n in names]), grid, names,
                 tags={"scene_id": meta["scene_id"], "datetime": meta["datetime"]})

    in_aoi = inside_aoi.sum()
    counts = {QA_LABELS[c]: int(((qa == c) & inside_aoi).sum()) for c in QA_LABELS if c != QA_OUTSIDE}
    stats = {"scene_id": meta["scene_id"], "datetime": meta["datetime"],
             "aoi_pixels": int(in_aoi), "valid_fraction": round(counts["valid"] / max(in_aoi, 1), 4),
             **{f"px_{k}": v for k, v in counts.items()},
             "median_ndvi_valid": float(np.nanmedian(np.where(qa == QA_VALID, ndvi, np.nan))) if counts["valid"] else None,
             "solar_zenith": meta.get("solar_zenith")}
    log.info("Scene %s: %.1f%% of AOI valid", meta["scene_id"], 100 * stats["valid_fraction"])
    return SceneResult(meta["scene_id"], meta["datetime"], layers, stats)
