"""Multi-date composite of per-scene absorption features.

For each feature, over the scenes where a pixel is valid (QA class 0):

    depth        inverse-variance weighted mean depth,  w_i = 1 / sigma_i^2
    depth_sigma  sqrt(1 / sum w_i)
    snr          depth / depth_sigma
    n_valid      scenes with a valid measurement
    n_detected   scenes where depth_i > k * sigma_i, depth_i > min_depth and the minimum
                 lies inside the search window (not on its edge)
    detected     1 where the composite passes the depth test and n_detected >= 1, else 0
    position     inverse-variance weighted mean position over detecting scenes (nm)
    position_sigma

Weighting assumes the ground did not change between dates, which holds for rock and
soil minerals but not for crops; vegetated pixels are excluded by the QA mask.
"""

from __future__ import annotations

import numpy as np

from ..config import SpectralConfig
from .features import FEATURES
from .scene import QA_VALID, SceneResult

COMPOSITE_BANDS = ("depth", "depth_sigma", "snr", "n_valid", "n_detected", "detected", "position", "position_sigma")


def composite(scenes: list[SceneResult], cfg: SpectralConfig) -> dict[str, dict[str, np.ndarray]]:
    k, floor = cfg.detect_sigma, cfg.min_depth
    out: dict[str, dict[str, np.ndarray]] = {}
    valid = np.stack([s.layers["qa"] == QA_VALID for s in scenes])

    for feat in FEATURES:
        d = np.stack([s.layers[f"{feat.key}_depth"] for s in scenes])
        sd = np.stack([s.layers[f"{feat.key}_depth_sigma"] for s in scenes])
        p = np.stack([s.layers[f"{feat.key}_position"] for s in scenes])
        sp = np.stack([s.layers[f"{feat.key}_position_sigma"] for s in scenes])

        ok = valid & np.isfinite(d) & np.isfinite(sd) & (sd > 0)
        w = np.where(ok, 1.0 / np.where(ok, sd, 1.0) ** 2, 0.0)
        wsum = w.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = np.where(wsum > 0, (w * np.nan_to_num(d)).sum(axis=0) / wsum, np.nan)
            sigma = np.where(wsum > 0, 1.0 / np.sqrt(wsum), np.nan)
            snr = depth / sigma

        # A scene detects the feature only if the minimum is bracketed inside the search window
        # (finite position); a minimum on the window edge belongs to a neighbouring feature.
        det_i = ok & (d > k * sd) & (d > floor) & np.isfinite(p)
        detected = (snr > k) & (depth > floor) & (det_i.sum(axis=0) >= 1)

        pok = det_i & np.isfinite(p) & np.isfinite(sp) & (sp > 0)
        pw = np.where(pok, 1.0 / np.where(pok, sp, 1.0) ** 2, 0.0)
        pwsum = pw.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pos = np.where(pwsum > 0, (pw * np.nan_to_num(p)).sum(axis=0) / pwsum, np.nan)
            pos_sigma = np.where(pwsum > 0, 1.0 / np.sqrt(pwsum), np.nan)

        n_valid = ok.sum(axis=0).astype("float32")
        no_obs = n_valid == 0
        out[feat.key] = {
            "depth": depth.astype("float32"),
            "depth_sigma": sigma.astype("float32"),
            "snr": snr.astype("float32"),
            "n_valid": np.where(no_obs, np.nan, n_valid).astype("float32"),
            "n_detected": np.where(no_obs, np.nan, det_i.sum(axis=0)).astype("float32"),
            "detected": np.where(no_obs, np.nan, detected).astype("float32"),
            "position": np.where(detected, pos, np.nan).astype("float32"),
            "position_sigma": np.where(detected, pos_sigma, np.nan).astype("float32"),
        }
    return out


def ancillary_composite(scenes: list[SceneResult]) -> dict[str, np.ndarray]:
    valid = np.stack([s.layers["qa"] == QA_VALID for s in scenes])
    ndvi_all = np.stack([s.layers["ndvi"] for s in scenes])
    cai_all = np.stack([s.layers["cai"] for s in scenes])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return {
                "n_valid_scenes": valid.sum(axis=0).astype("float32"),
                "ndvi_median": np.nanmedian(ndvi_all, axis=0).astype("float32"),
                "ndvi_min": np.nanmin(ndvi_all, axis=0).astype("float32"),
                "cai_median_valid": np.nanmedian(np.where(valid, cai_all, np.nan), axis=0).astype("float32"),
            }
