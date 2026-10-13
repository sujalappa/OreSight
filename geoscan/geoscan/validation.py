"""Validation of prospectivity surfaces against known mineralisation.

All metrics compare a score surface with ground truth inside the AOI:

* ``auc``            Mann-Whitney AUC: probability that a random positive scores higher than a
                     random negative (0.5 = no skill, 1.0 = perfect).
* ``bootstrap_auc``  stratified bootstrap confidence interval and one-sided permutation p-value.
* ``success_rate``   share of mineralised area captured when the AOI is ranked by score
                     (Agterberg & Bonham-Carter 2005 style success-rate curve).
* ``target_hits``    how many ranked targets fall within a distance of a lode, versus the number
                     expected if the same number of points were placed at random in the AOI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not pos.size or not neg.size:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def bootstrap_auc(pos: np.ndarray, neg: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    rng = np.random.default_rng(seed)
    a = auc(pos, neg)
    boots = [auc(rng.choice(pos, pos.size), rng.choice(neg, neg.size)) for _ in range(n)]
    allv = np.concatenate([pos, neg])
    perm = []
    for _ in range(n):
        s = rng.permutation(allv)
        perm.append(auc(s[:pos.size], s[pos.size:]))
    return {"auc": round(a, 3), "ci_low": round(float(np.nanpercentile(boots, 2.5)), 3),
            "ci_high": round(float(np.nanpercentile(boots, 97.5)), 3),
            "p_value": round(float((np.sum(np.array(perm) >= a) + 1) / (n + 1)), 4),
            "n_pos": int(pos.size), "n_neg": int(neg.size)}


def success_rate(score: np.ndarray, positive: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    """Cumulative share of positive cells vs share of AOI area, ranking cells by score (high first)."""
    s, p = score[valid], positive[valid]
    ok = np.isfinite(s)
    s, p = s[ok], p[ok]
    order = np.argsort(-s, kind="stable")
    cum_pos = np.cumsum(p[order]) / max(p.sum(), 1)
    area = np.arange(1, s.size + 1) / s.size
    return pd.DataFrame({"area_fraction": area, "captured_fraction": cum_pos})


def captured_at(curve: pd.DataFrame, fractions=(0.1, 0.2, 0.3)) -> dict:
    return {f"captured_top{int(100 * f)}pct": round(float(np.interp(f, curve["area_fraction"],
                                                                     curve["captured_fraction"])), 3)
            for f in fractions}


def target_hits(target_xy: np.ndarray, lodes_utm, aoi_utm, radii=(100.0, 200.0)) -> dict:
    from shapely.geometry import Point
    union = lodes_utm.union_all() if hasattr(lodes_utm, "union_all") else lodes_utm.unary_union
    out = {"n_targets": int(len(target_xy))}
    if not len(target_xy):
        return out
    d = np.array([union.distance(Point(x, y)) for x, y in target_xy])
    out["median_distance_to_lode_m"] = round(float(np.median(d)), 1)
    for r in radii:
        frac_area = union.buffer(r).intersection(aoi_utm).area / aoi_utm.area
        out[f"hits_within_{int(r)}m"] = int((d <= r).sum())
        out[f"expected_random_{int(r)}m"] = round(float(frac_area * len(target_xy)), 1)
    return out


def zone_capture(zone_masks: dict[str, np.ndarray], truth: np.ndarray, inside: np.ndarray) -> pd.DataFrame:
    """How much of the ground truth falls inside each set of zones, against the area the zones cover.

    zone_masks: label -> boolean raster (e.g. "Tier 1 zones", "Tier 1 + 2 zones", ...).
    enrichment = share of truth captured / share of the block covered (1.0 = no better than chance).
    """
    n_in, n_truth = int(inside.sum()), int((truth & inside).sum())
    rows = []
    for name, m in zone_masks.items():
        m = m & inside
        area, cap = m.sum() / max(n_in, 1), (m & truth).sum() / max(n_truth, 1)
        rows.append({"zones": name, "pct_of_block": round(100 * float(area), 1),
                     "pct_of_truth_captured": round(100 * float(cap), 1),
                     "enrichment": round(float(cap / area), 1) if area > 0 else None})
    return pd.DataFrame(rows)
