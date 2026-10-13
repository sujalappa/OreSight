"""Copper Prospectivity Index (CPI), prospective zones and suggested field-check locations
(methodology sections 5-6).

    H   = 0.30 a_musc + 0.28 a_qtz + 0.22 a_goss + 0.20 a_chl
    CPI = 1_block * (0.70 H + 0.30 S_prox),     S_prox = exp(-d_lineament / 250 m)

CPI is not re-stretched, so the tier thresholds (0.70 / 0.50 / 0.28) are absolute.

The result of this kind of work is a *region*, not a point. The data are 60-90 m pixels spread over a
few hundred metres, and they describe the surface, not the ore. So the main product is the set of
**prospective zones**: connected areas with CPI >= the Tier 2 threshold (plus isolated Tier 3 anomalies),
ranked by tier and then by peak CPI. The Tier 3 contour around them is their outer halo.

**Suggested field-check locations** are a convenience for planning traverses: the highest-CPI cells,
kept >= 250 m apart so each anomaly gets about one. They mark where to start looking inside a zone.
They are not drill collars, and their exact position is not significant.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.features import shapes
from scipy.ndimage import label
from shapely.geometry import shape
from shapely.ops import unary_union

from ..config import TargetConfig
from ..grid import Grid

MINERALS = ("muscovite", "quartz", "gossan", "chlorite")
TIER_LABELS = {1: "Tier 1: High Priority", 2: "Tier 2: Medium Priority", 3: "Tier 3: General Anomaly"}
FOLLOW_UP = {
    1: "Detailed geological mapping, systematic rock-chip / soil sampling and ground geophysics",
    2: "Reconnaissance mapping and rock-chip sampling",
    3: "Low priority: check during regional traverses",
    0: "Field verification",
}


def cpi(a: dict[str, np.ndarray], s_prox: np.ndarray, inside: np.ndarray, cfg: TargetConfig):
    h = (cfg.w_muscovite * a["muscovite"] + cfg.w_quartz * a["quartz"] +
         cfg.w_gossan * a["gossan"] + cfg.w_chlorite * a["chlorite"])
    c = cfg.w_hydrothermal * h + cfg.w_structure * s_prox
    return np.where(inside, c, np.nan).astype("float32"), np.where(inside, h, np.nan).astype("float32")


def tier_number(value: float, cfg: TargetConfig) -> int:
    return 1 if value >= cfg.tier1 else 2 if value >= cfg.tier2 else 3 if value >= cfg.tier3 else 0


def tier(value: float, cfg: TargetConfig) -> tuple[str, str]:
    """(tier label, suggested follow-up). Follow-up is staged field work; nothing here justifies drilling."""
    n = tier_number(value, cfg)
    return TIER_LABELS.get(n, "Below Tier 3"), FOLLOW_UP[n]


def location_name(rank: int) -> str:
    if rank <= 26:
        return f"Location {chr(64 + rank)}"
    return f"Location A{chr(64 + rank - 26)}"


def prospective_zones(c: np.ndarray, grid: Grid, a: dict[str, np.ndarray], s_prox: np.ndarray, cfg: TargetConfig,
                      lineaments: gpd.GeoDataFrame | None = None, support: np.ndarray | None = None,
                      min_support: float = 0.20):
    """Ranked prospective zones.

    A zone is a connected area with CPI >= tier2 (Tier 1 if it reaches tier1 somewhere, otherwise Tier 2).
    A weak anomaly that is separate from every such zone and only reaches tier3 is a Tier 3 zone.
    The tier3 contour around the Tier 1 / 2 zones is their outer halo; it is not part of the zone.
    (Using tier3 as the boundary merges whole districts into one blob: at Thanewasna it covered 56 % of the
    block for 74 % of the lode area, against 16.5 % of the block for 48.5 % of the lodes with tier2.)

    Returns (GeoDataFrame in grid CRS, int raster with the zone rank per cell, 0 = no zone).
    """
    eight = np.ones((3, 3), int)
    finite = np.isfinite(c)
    core, n_core = label(finite & (c >= cfg.tier2), structure=eight)
    halo, n_halo = label(finite & (c >= cfg.tier3), structure=eight)
    masks = [core == k for k in range(1, n_core + 1)]
    with_core = set(np.unique(halo[core > 0]))                       # halo components that contain a core
    masks += [halo == k for k in range(1, n_halo + 1) if k not in with_core]

    cell_km2 = grid.res ** 2 / 1e6
    ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    weights = {"muscovite": cfg.w_muscovite, "quartz": cfg.w_quartz, "gossan": cfg.w_gossan, "chlorite": cfg.w_chlorite}
    lines = unary_union(list(lineaments.geometry)) if lineaments is not None and len(lineaments) else None

    recs, geoms, kept = [], [], []
    for m in masks:
        area = float(m.sum()) * cell_km2
        if area < cfg.zone_min_area_km2:
            continue
        geom = unary_union([shape(g) for g, v in shapes(m.astype("uint8"), mask=m, transform=grid.transform,
                                                         connectivity=8)])
        geom = geom.buffer(grid.res / 2).buffer(-grid.res / 2).simplify(grid.res / 4)   # soften the pixel steps
        if geom.is_empty:
            continue
        cz = c[m]
        peak = float(cz.max())
        means = {k2: float(np.nanmean(a[k2][m])) for k2 in MINERALS}
        contrib = sorted(((weights[k2] * means[k2], k2) for k2 in MINERALS), reverse=True)
        struct = float(np.nanmean(s_prox[m]))
        evidence = [k2 for _, k2 in contrib[:2]] + (["structure"] if struct >= 0.5 else [])
        pr, pc = np.unravel_index(np.argmax(np.where(m, c, -1)), c.shape)          # the peak cell
        lon, lat = ll.transform(grid.x0 + (pc + 0.5) * grid.res, grid.y0 - (pr + 0.5) * grid.res)
        t = tier_number(peak, cfg)
        # same rule as the map hatching: a cell is assessed if EMIT saw the ground there OR detected a mineral
        emit = np.maximum.reduce([a["muscovite"][m], a["chlorite"][m], a["gossan"][m]])
        seen = 1.0 if support is None else float(((np.nan_to_num(support[m]) >= min_support) | (emit >= 0.05)).mean())
        basis = "EMIT alteration, quartz and structure" if seen >= 0.8 else \
            "Partly assessed by EMIT; elsewhere quartz and structure only" if seen >= 0.2 else \
            "Quartz and structure only (EMIT not assessed)"
        recs.append({
            "tier": t, "Priority_Tier": TIER_LABELS[t], "Area_km2": round(area, 3),
            "CPI_Max_Pct": round(100 * peak, 1), "CPI_Mean_Pct": round(100 * float(cz.mean()), 1),
            "Tier1_Area_km2": round(float((cz >= cfg.tier1).sum()) * cell_km2, 3),
            **{f"{k2.capitalize()}_Pct": round(100 * means[k2], 1) for k2 in MINERALS},
            "Structural_Proximity": round(struct, 2),
            "Lineaments_km": round(geom.intersection(lines).length / 1000, 2) if lines is not None else 0.0,
            "Main_Evidence": ", ".join(evidence), "Evidence_Basis": basis,
            "Peak_Latitude": round(float(lat), 4), "Peak_Longitude": round(float(lon), 4),
            "Suggested_Follow_Up": FOLLOW_UP[t],
        })
        geoms.append(geom)
        kept.append(m)

    order = sorted(range(len(recs)), key=lambda i: (recs[i]["tier"], -recs[i]["CPI_Max_Pct"], -recs[i]["Area_km2"]))
    rank_raster = np.zeros(c.shape, "int32")
    out = []
    for rank, i in enumerate(order, start=1):
        rank_raster[kept[i]] = rank
        out.append({"Zone": rank, "Zone_ID": f"Z{rank:02d}", **{k: v for k, v in recs[i].items() if k != "tier"}})
    gdf = gpd.GeoDataFrame(out, geometry=[geoms[i] for i in order], crs=grid.crs)
    return gdf, rank_raster


def select_locations(c: np.ndarray, grid: Grid, layers: dict[str, np.ndarray], eligible: np.ndarray,
                     cfg: TargetConfig, zone_rank: np.ndarray | None = None) -> pd.DataFrame:
    """Suggested field-check locations: highest-CPI cells, greedily kept >= min_spacing_m apart."""
    ok = eligible & np.isfinite(c) & (c >= cfg.min_cpi)
    rows, cols = np.nonzero(ok)
    order = np.argsort(-c[rows, cols], kind="stable")
    rows, cols = rows[order], cols[order]
    xs = grid.x0 + (cols + 0.5) * grid.res
    ys = grid.y0 - (rows + 0.5) * grid.res

    chosen: list[int] = []
    cx, cy = np.empty(0), np.empty(0)
    for i in range(len(rows)):
        if cx.size and np.min((cx - xs[i]) ** 2 + (cy - ys[i]) ** 2) < cfg.min_spacing_m ** 2:
            continue
        chosen.append(i)
        cx, cy = np.append(cx, xs[i]), np.append(cy, ys[i])
        if len(chosen) >= cfg.n_targets:
            break

    lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(xs[chosen], ys[chosen])
    recs = []
    for rank, (i, lo, la) in enumerate(zip(chosen, lon, lat), start=1):
        r, cc = rows[i], cols[i]
        t, follow = tier(float(c[r, cc]), cfg)
        z = int(zone_rank[r, cc]) if zone_rank is not None else 0
        where = f"Z{z:02d}" if z else "outer halo" if c[r, cc] >= cfg.tier3 else "outside zones"
        row = {"Rank": rank, "Location_ID": f"FC-{rank:02d}", "Location_Name": location_name(rank),
               "Zone": where,
               "Latitude": round(float(la), 5), "Longitude": round(float(lo), 5),
               "Easting": round(float(xs[i]), 1), "Northing": round(float(ys[i]), 1),
               "CPI": round(float(c[r, cc]), 4), "CPI_Percent": round(100 * float(c[r, cc]), 1),
               "Priority_Tier": t}
        for k, v in layers.items():
            row[k] = round(float(v[r, cc]), 4) if np.isfinite(v[r, cc]) else None
        row["Check_Within_m"] = int(round(cfg.min_spacing_m))
        row["Suggested_Follow_Up"] = follow
        recs.append(row)
    return pd.DataFrame(recs)


# backwards-compatible name used by older scripts
select_targets = select_locations
