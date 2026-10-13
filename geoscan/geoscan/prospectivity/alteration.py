"""Four alteration layers (methodology sections 3-4): muscovite, chlorite, gossan, quartz.

Evidence per EMIT pixel (60 m, multi-date composite from ``emit_l2a``):

    muscovite / sericite (phyllic)   Al-OH ~2200 nm detected, minimum at 2190-2215 nm
    chlorite / epidote (propylitic)  Fe-OH ~2250 nm detected, OR Mg-OH ~2330 nm detected where
                                     EMIT L2B does not identify carbonate more often than
                                     chlorite/epidote/prehnite/amphibole (calcite also absorbs
                                     at ~2340 nm)
    gossan / iron oxide              Fe3+ ~900 nm detected (minimum 850-965 nm)

Each detection carries its band depth BD and weight w = 1 / sigma_BD (the L2A analogue of the
methodology's fit / uncertainty), normalised so the mean weight of detections is 1.

Two-tier hydrothermal potential field (section 4.3), G = Gaussian kernel:

    F_core   = G_core * (w BD) / G_core * w                 sigma_core = 90 m
    W_halo   = G_halo * (G_core * w)                        sigma_halo = 375 m
    conf     = 1 - exp(-W_halo / theta)                     theta = 0.05
    F        = (G_core * F_core) * conf
    a        = clip(F / P98(F > 0 in AOI), 0, 1)            (section 4.4)

With the normalised weights, W_halo is the kernel-weighted fraction of detected pixels, so
theta = 0.05 means confidence 63 % where ~5 % of the surrounding pixels show the absorption.

Quartz (section 4.5): QSI from ASTER GED, a_qtz = clip((QSI - P15) / (P98 - P15), 0, 1).

``support`` is the halo-weighted fraction of usable (unmasked) EMIT pixels. Where it is low
the EMIT layers carry no information (vegetation, cloud), which maps show as "no data".
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ..config import AlterationConfig

MINERALS = ("muscovite", "chlorite", "gossan", "quartz")
LABELS = {
    "muscovite": "Muscovite / Sericite (Phyllic)",
    "chlorite": "Chlorite / Epidote (Propylitic)",
    "gossan": "Gossan / Iron Oxide (Fe³⁺)",
    "quartz": "Quartz / Silicification",
}


# Zone tiers used for vectorised anomaly polygons (same thresholds as the legacy outputs).
ZONE_TIERS = {
    "muscovite": ((0.30, 0.55, 0.75, 1.001), ("Moderate Phyllic", "Strong Phyllic", "High-Grade Sericite Core")),
    "quartz": ((0.35, 0.60, 0.80, 1.001), ("Moderate Silicification", "Strong Quartz Flooding", "Massive Quartz Reef")),
    "gossan": ((0.25, 0.50, 0.70, 1.001), ("Weak Oxidation", "Moderate Gossan", "Intense Iron Cap")),
    "chlorite": ((0.30, 0.50, 0.70, 1.001), ("Outer Propylitic", "Moderate Chlorite", "Dense Propylitic Halo")),
}
CONTOUR_LEVELS = {"muscovite": (30, 55, 75), "quartz": (35, 60, 80), "gossan": (25, 50, 70), "chlorite": (30, 50, 70)}


def potential_field(depth: np.ndarray, weight: np.ndarray, res: float, cfg: AlterationConfig) -> np.ndarray:
    """Two-tier field. ``depth``/``weight`` are 0 where there is no detection."""
    s_core, s_halo = cfg.sigma_core_m / res, cfg.sigma_halo_m / res
    w = np.nan_to_num(weight)
    wd = w * np.nan_to_num(depth)
    num = gaussian_filter(wd, s_core, mode="constant")
    den = gaussian_filter(w, s_core, mode="constant")
    core = np.where(den > 1e-3, num / np.maximum(den, 1e-12), 0.0)
    w_halo = gaussian_filter(den, s_halo, mode="constant")
    conf = 1.0 - np.exp(-w_halo / cfg.theta_decay)
    return (gaussian_filter(core, s_core, mode="constant") * conf).astype("float32")


def normalise(field: np.ndarray, inside: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    vals = field[inside & (field > 0)]
    ref = float(np.percentile(vals, percentile)) if vals.size else 0.0
    a = np.clip(field / ref, 0.0, 1.0) if ref > 0 else np.zeros_like(field)
    return np.where(inside, a, np.nan).astype("float32"), ref


def _evidence(layers: dict, key: str, mask: np.ndarray | None = None):
    """Depth and weight of detections of one feature (0 elsewhere)."""
    L = layers[key]
    det = (L["detected"] == 1)
    if mask is not None:
        det &= mask
    w = np.where(det, 1.0 / np.maximum(L["depth_sigma"], 1e-4), 0.0)
    d = np.where(det, L["depth"], 0.0)
    return d, w, det


def emit_evidence(features: dict[str, dict[str, np.ndarray]], l2b_n_dates: dict[str, np.ndarray] | None,
                  cfg: AlterationConfig) -> dict[str, dict[str, np.ndarray]]:
    """Per-mineral detection depth, weight and boolean mask on the EMIT grid."""
    lo, hi = cfg.sericite_position_nm
    pos = features["aloh_2200"]["position"]
    musc_d, musc_w, musc_det = _evidence(features, "aloh_2200", (pos >= lo) & (pos <= hi))

    not_carbonate = np.ones_like(pos, dtype=bool)
    if l2b_n_dates is not None and "carbonate" in l2b_n_dates:
        carb = np.nan_to_num(l2b_n_dates["carbonate"])
        prop = sum(np.nan_to_num(l2b_n_dates[k]) for k in cfg.propylitic_l2b_classes if k in l2b_n_dates)
        not_carbonate = ~(carb > prop)
    fe_d, fe_w, fe_det = _evidence(features, "feoh_2250")
    mg_d, mg_w, mg_det = _evidence(features, "mgoh_2330", not_carbonate)
    # A pixel may show both: keep the stronger (higher-weight) measurement.
    use_fe = fe_w >= mg_w
    chl_d = np.where(use_fe, fe_d, mg_d)
    chl_w = np.where(use_fe, fe_w, mg_w)
    chl_det = fe_det | mg_det

    gos_d, gos_w, gos_det = _evidence(features, "ferric_900")

    out = {}
    for name, d, w, det in (("muscovite", musc_d, musc_w, musc_det), ("chlorite", chl_d, chl_w, chl_det),
                            ("gossan", gos_d, gos_w, gos_det)):
        mean_w = w[det].mean() if det.any() else 1.0
        out[name] = {"depth": d, "weight": w / mean_w, "detected": det}
    return out


def add_l2b_evidence(evidence: dict, l2b_weight: dict, l2b_depth: dict, l2a_valid: np.ndarray,
                     cfg: AlterationConfig) -> dict:
    """Merge L2B identifications (weight = fit / uncertainty) into the L2A evidence.

    Both weight sets are normalised to mean 1 over their own detections, so neither source
    dominates by units. Where both exist, depth is the weight-averaged depth and weights add.
    """
    if cfg.l2b_evidence == "none":
        return evidence
    out = {}
    for m, ev in evidence.items():
        classes = [k for k in cfg.l2b_classes.get(m, []) if k in l2b_weight]
        wb = sum(np.nan_to_num(l2b_weight[k]) for k in classes) if classes else np.zeros_like(ev["weight"])
        db = sum(np.nan_to_num(l2b_weight[k]) * np.nan_to_num(l2b_depth[k]) for k in classes) \
            if classes else np.zeros_like(ev["weight"])
        db = np.where(wb > 0, db / np.maximum(wb, 1e-12), 0.0)
        if cfg.l2b_evidence == "combine_valid":
            wb = np.where(l2a_valid, wb, 0.0)
        det_b = wb > 0
        wb = wb / (wb[det_b].mean() if det_b.any() else 1.0)
        wa, da = ev["weight"], ev["depth"]
        w = wa + wb
        d = np.where(w > 0, (wa * da + wb * db) / np.maximum(w, 1e-12), 0.0)
        out[m] = {"depth": d, "weight": w, "detected": ev["detected"] | det_b}
    return out


def unit_raster(lith_utm, field: str, grid) -> tuple[np.ndarray, dict[int, str]]:
    """Integer rock-unit codes on ``grid`` (0 = unmapped). Larger units are burnt first so small
    bodies drawn on top of them keep their own code."""
    from rasterio.features import rasterize
    names = lith_utm[field].astype(str).str.lower().str.split().str.join(" ")
    order = lith_utm.assign(_u=names, _a=lith_utm.area).sort_values("_a", ascending=False)
    codes = {u: i + 1 for i, u in enumerate(dict.fromkeys(order["_u"]))}
    out = np.zeros(grid.shape, dtype="int32")
    for _, row in order.iterrows():
        m = rasterize([(row.geometry, 1)], out_shape=grid.shape, transform=grid.transform, dtype="uint8").astype(bool)
        out[m] = codes[row["_u"]]
    return out, {v: k for k, v in codes.items()}


def lithology_background(field: np.ndarray, units: np.ndarray, valid: np.ndarray, min_pixels: int) -> tuple:
    """Per-unit median background removal. Returns (positive residual, {unit code: background})."""
    v = valid & np.isfinite(field)
    global_bg = float(np.median(field[v])) if v.any() else 0.0
    out = np.array(field, dtype="float32")
    bgs = {}
    for u in np.unique(units):
        m = units == u
        mv = m & v
        bg = float(np.median(field[mv])) if (u != 0 and mv.sum() >= min_pixels) else global_bg
        out[m] = field[m] - bg
        bgs[int(u)] = bg
    return np.clip(out, 0.0, None), bgs


def data_support(n_valid: np.ndarray, res: float, cfg: AlterationConfig) -> np.ndarray:
    usable = (np.nan_to_num(n_valid) > 0).astype("float64")
    sig = cfg.support_sigma_m
    if sig == 0:                                   # local: the pixel itself or one of its 8 neighbours was seen
        from scipy.ndimage import maximum_filter
        return maximum_filter(usable, size=3, mode="constant").astype("float32")
    sigma_px = (cfg.sigma_halo_m / 2 if sig is None else sig) / res
    return gaussian_filter(usable, sigma_px, mode="constant").astype("float32")


def quartz_layer(qsi: np.ndarray, inside: np.ndarray, cfg: AlterationConfig) -> np.ndarray:
    v = qsi[inside & np.isfinite(qsi)]
    bg, p98 = np.percentile(v, [cfg.quartz_background_percentile, cfg.reference_percentile])
    return np.where(inside, np.clip((qsi - bg) / (p98 - bg + 1e-12), 0.0, 1.0), np.nan).astype("float32")
