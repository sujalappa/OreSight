"""Topographic lineaments from a DEM.

Pipeline (all distances in metres on a UTM grid):

1. Gaussian-smoothed DEM -> slope/aspect -> hillshade from ``n_azimuths`` directions,
   each locally contrast-equalised (CLAHE) so escarpments do not dominate the thresholds.
2. Canny edges on each hillshade with quantile thresholds, plus the axes of linear valleys
   and ridges from a Hessian (Sato) filter on the DEM (illumination independent).
3. Probabilistic Hough transform on the union of edge maps.
4. Collinear merging (angle / cross-track / along-track gap tolerances), then removal of
   shorter near-duplicates, then a minimum-length filter.
5. Per-line attributes: strike, edge coverage, illumination / valley / ridge support, and a
   cross-profile morphology class: valley, ridge or scarp (one side systematically higher).
6. Rasters on the analysis grid: distance to nearest lineament and lineament density. By
   default scarps are excluded from these (erosional escarpment edges, e.g. Deccan plateau
   margins, are not fluid pathways); they remain in the vector output.

The DEM is processed on a grid padded beyond the AOI, so lines and distances near the
AOI boundary are not truncated. Lineaments are topographic: they include faults and
fractures but also drainage, ridges and escarpments, and need geological checking.
"""

from __future__ import annotations

import logging

import cv2
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, map_coordinates
from scipy.signal import fftconvolve
from shapely.geometry import LineString
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from skimage.filters import sato
from skimage.morphology import skeletonize

from ..config import StructureConfig
from ..grid import Grid

log = logging.getLogger(__name__)


def terrain(dem: np.ndarray, res: float, smooth_px: float):
    """Slope (rad) and aspect (rad, clockwise from north, direction the slope faces)."""
    z = gaussian_filter(dem, smooth_px) if smooth_px > 0 else dem
    dz_drow, dz_dcol = np.gradient(z, res)
    dz_dx, dz_dy = dz_dcol, -dz_drow        # east, north (row 0 is north)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, -dz_dy)
    return slope, aspect


def hillshade(slope, aspect, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az, zen = np.radians(azimuth_deg), np.radians(90.0 - elevation_deg)
    return np.clip(np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(az - aspect), 0, 1)


def _strike(x1, y1, x2, y2):
    return np.degrees(np.arctan2(x2 - x1, y2 - y1)) % 180.0


def _angle_diff(a, b):
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def _perp(px, py, x1, y1, ux, uy):
    return np.abs((px - x1) * uy - (py - y1) * ux)


def merge_collinear(seg: np.ndarray, angle_tol: float, perp_tol: float, gap_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """seg: (n, 4) x1,y1,x2,y2. Returns merged segments and how many raw segments each contains."""
    count = np.ones(len(seg), dtype=int)
    changed = True
    while changed and len(seg) > 1:
        changed = False
        length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        order = np.argsort(-length)
        seg, count, length = seg[order], count[order], length[order]
        az = _strike(*seg.T)
        ux, uy = (seg[:, 2] - seg[:, 0]) / length, (seg[:, 3] - seg[:, 1]) / length
        mx, my = (seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2
        used = np.zeros(len(seg), bool)
        new_seg, new_count = [], []
        for i in range(len(seg)):
            if used[i]:
                continue
            cand = ~used
            cand[i] = False
            cand &= _angle_diff(az, az[i]) <= angle_tol
            j = np.nonzero(cand)[0]
            if j.size:
                pd_ = np.minimum(_perp(mx[j], my[j], seg[i, 0], seg[i, 1], ux[i], uy[i]),
                                 _perp(mx[i], my[i], seg[j, 0], seg[j, 1], ux[j], uy[j]))
                keep = pd_ <= perp_tol
                j, pd_ = j[keep], pd_[keep]
            if j.size:
                s_i = np.array([0.0, length[i]])
                p1 = (seg[j, 0] - seg[i, 0]) * ux[i] + (seg[j, 1] - seg[i, 1]) * uy[i]
                p2 = (seg[j, 2] - seg[i, 0]) * ux[i] + (seg[j, 3] - seg[i, 1]) * uy[i]
                gap = np.maximum(np.minimum(p1, p2) - s_i[1], s_i[0] - np.maximum(p1, p2))
                keep = gap <= gap_tol
                j, pd_, gap = j[keep], pd_[keep], gap[keep]
            if j.size:
                b = j[np.argmin(pd_ + 0.5 * np.maximum(gap, 0))]
                pts = np.array([seg[i, :2], seg[i, 2:], seg[b, :2], seg[b, 2:]])
                d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
                a, c = np.unravel_index(np.argmax(d), d.shape)
                new_seg.append(np.concatenate([pts[a], pts[c]]))
                new_count.append(count[i] + count[b])
                used[i] = used[b] = True
                changed = True
            else:
                new_seg.append(seg[i])
                new_count.append(count[i])
                used[i] = True
        seg, count = np.array(new_seg), np.array(new_count)
    return seg, count


def remove_duplicates(seg: np.ndarray, count: np.ndarray, angle_tol: float, perp_tol: float):
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    order = np.argsort(-length)
    seg, count, length = seg[order], count[order], length[order]
    az = _strike(*seg.T)
    mx, my = (seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2
    keep = np.ones(len(seg), bool)
    for i in range(len(seg)):
        if not keep[i]:
            continue
        ux, uy = (seg[i, 2] - seg[i, 0]) / length[i], (seg[i, 3] - seg[i, 1]) / length[i]
        j = np.arange(i + 1, len(seg))
        j = j[keep[j] & (_angle_diff(az[j], az[i]) <= angle_tol)]
        if not j.size:
            continue
        cross = _perp(mx[j], my[j], seg[i, 0], seg[i, 1], ux, uy)
        along = (mx[j] - seg[i, 0]) * ux + (my[j] - seg[i, 1]) * uy
        dup = (cross <= perp_tol) & (along >= -0.1 * length[i]) & (along <= 1.1 * length[i])
        keep[j[dup]] = False
    return seg[keep], count[keep]


def _sample_pixels(line: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = line
    n = max(2, int(np.hypot(x2 - x1, y2 - y1) / (grid.res / 2)))
    xs, ys = np.linspace(x1, x2, n), np.linspace(y1, y2, n)
    cols = np.clip(((xs - grid.x0) / grid.res).astype(int), 0, grid.width - 1)
    rows = np.clip(((grid.y0 - ys) / grid.res).astype(int), 0, grid.height - 1)
    return rows, cols


def strike_class(az: float) -> str:
    if az < 22.5 or az >= 157.5:
        return "N-S"
    if az < 67.5:
        return "NE-SW"
    if az < 112.5:
        return "E-W"
    return "NW-SE"


def hillshade_edges(slope, aspect, grid: Grid, cfg: StructureConfig) -> np.ndarray:
    """Canny edges of locally contrast-equalised hillshades, one map per illumination azimuth.

    CLAHE (adaptive histogram equalisation over ~``clahe_window_m`` tiles) stops high-relief
    areas (escarpments) from monopolising the quantile thresholds, so subtle features in flat
    ground are detected with the same sensitivity.
    """
    kernel = max(8, int(round(cfg.clahe_window_m / grid.res)))
    out = []
    for az in np.arange(cfg.n_azimuths) * (360.0 / cfg.n_azimuths):
        hs = hillshade(slope, aspect, az, cfg.sun_elevation_deg)
        if cfg.local_contrast:
            hs = equalize_adapthist(hs, kernel_size=kernel, clip_limit=cfg.clahe_clip)
        out.append(tiled_canny(hs, grid.res, 1.0, cfg, cfg.canny_tile_m))
    return np.stack(out)


def tiled_canny(img: np.ndarray, res: float, sigma: float, cfg: StructureConfig, tile_m: float) -> np.ndarray:
    """Canny with quantile thresholds. With ``tile_m`` > 0 the quantiles are computed per tile
    (with 25 % overlap) so flat ground is not out-competed by high-relief ground in the same grid."""
    def run(x):
        return canny(x, sigma=sigma, low_threshold=cfg.canny_low_quantile,
                     high_threshold=cfg.canny_high_quantile, use_quantiles=True)

    if tile_m <= 0:
        return run(img)
    T = max(32, int(tile_m / res))
    P = T // 4
    out = np.zeros(img.shape, bool)
    for r0 in range(0, img.shape[0], T):
        for c0 in range(0, img.shape[1], T):
            ra, ca = max(0, r0 - P), max(0, c0 - P)
            e = run(img[ra:r0 + T + P, ca:c0 + T + P])
            out[r0:r0 + T, c0:c0 + T] = e[r0 - ra:r0 - ra + T, c0 - ca:c0 - ca + T]
    return out


def ridge_valley_edges(dem: np.ndarray, grid: Grid, cfg: StructureConfig) -> np.ndarray:
    """Axes of linear valleys and ridges from the DEM Hessian (Sato tubeness filter).

    Illumination independent. Responses are normalised by their local mean (window
    ``clahe_window_m``) and thresholded at a quantile, then thinned to 1-pixel axes.
    Returns (2, rows, cols): valleys, ridges.
    """
    sig = [s / grid.res for s in cfg.ridge_sigmas_m]
    win = cfg.clahe_window_m / grid.res / 2
    maps = []
    for black in (True, False):          # True: valleys (dark ridges of elevation), False: ridges
        r = sato(dem, sigmas=sig, black_ridges=black, mode="reflect")
        r_norm = r / (gaussian_filter(r, win) + 1e-9) if cfg.local_contrast else r
        mask = r_norm > np.quantile(r_norm, cfg.ridge_quantile)
        maps.append(skeletonize(mask))
    return np.stack(maps)


def cross_profile(dem_s: np.ndarray, line: np.ndarray, grid: Grid, offset_m: float):
    """Median centre-minus-sides (sym) and half side difference (step) along a line, in metres."""
    x1, y1, x2, y2 = line
    L = np.hypot(x2 - x1, y2 - y1)
    n = max(3, int(L / grid.res))
    t = np.linspace(0.05, 0.95, n)
    xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    nx, ny = -(y2 - y1) / L, (x2 - x1) / L         # unit normal

    def z(x, y):
        return map_coordinates(dem_s, [(grid.y0 - y) / grid.res - 0.5, (x - grid.x0) / grid.res - 0.5],
                               order=1, mode="nearest")

    zc, zl, zr = z(xs, ys), z(xs + offset_m * nx, ys + offset_m * ny), z(xs - offset_m * nx, ys - offset_m * ny)
    sym = np.median(zc - 0.5 * (zl + zr))
    step = np.median(np.abs(zl - zr)) / 2 if np.all(np.sign(zl - zr) == np.sign(np.median(zl - zr))) \
        else abs(np.median(zl - zr)) / 2
    return float(sym), float(step)


def profile_class(sym: float, step: float, cfg: StructureConfig) -> str:
    """'scarp' when one side is systematically higher (one-sided step), else 'valley' / 'ridge'."""
    if step >= cfg.scarp_min_step_m and step > cfg.scarp_step_ratio * abs(sym):
        return "scarp"
    return "valley" if sym < 0 else "ridge"


def extract_lineaments(dem: np.ndarray, grid: Grid, aoi_geom_utm, cfg: StructureConfig):
    """dem on ``grid`` (padded). Returns (GeoDataFrame in grid CRS, dict of padded rasters)."""
    dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    slope, aspect = terrain(dem, grid.res, cfg.smooth_sigma_px)
    dem_s = gaussian_filter(dem, cfg.smooth_sigma_px)

    edges = hillshade_edges(slope, aspect, grid, cfg)
    rv = ridge_valley_edges(dem_s, grid, cfg) if cfg.ridge_valley else np.zeros((0,) + dem.shape, bool)
    union = edges.any(axis=0) | (rv.any(axis=0) if len(rv) else False)

    lines = cv2.HoughLinesP(union.astype(np.uint8) * 255, 1, np.pi / 180, cfg.hough_threshold,
                            minLineLength=int(round(cfg.min_segment_m / grid.res)),
                            maxLineGap=int(round(cfg.max_gap_m / grid.res)))
    raw = np.zeros((0, 4)) if lines is None else lines[:, 0, :].astype(float)
    # pixel (col,row) -> map (x,y) at pixel centres
    seg = np.column_stack([grid.x0 + (raw[:, 0] + 0.5) * grid.res, grid.y0 - (raw[:, 1] + 0.5) * grid.res,
                           grid.x0 + (raw[:, 2] + 0.5) * grid.res, grid.y0 - (raw[:, 3] + 0.5) * grid.res])
    seg = seg[np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= cfg.min_segment_m]
    n_raw = len(seg)

    seg, count = merge_collinear(seg, cfg.merge_angle_deg, cfg.merge_perp_m, cfg.merge_gap_m)
    n_merged = len(seg)
    seg, count = remove_duplicates(seg, count, cfg.dedup_angle_deg, cfg.dedup_perp_m)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    keep = length >= cfg.min_length_m
    seg, count, length = seg[keep], count[keep], length[keep]
    log.info("Lineaments: %d Hough segments -> %d merged -> %d final (>= %.0f m)",
             n_raw, n_merged, len(seg), cfg.min_length_m)

    edges_d = np.stack([binary_dilation(e, iterations=1) for e in edges])
    rv_d = np.stack([binary_dilation(e, iterations=1) for e in rv]) if len(rv) else rv
    union_d = binary_dilation(union, iterations=1)
    records, geoms = [], []
    for k, (line, n, L) in enumerate(zip(seg, count, length)):
        geom = LineString([line[:2], line[2:]])
        if not geom.intersects(aoi_geom_utm):
            continue
        rows, cols = _sample_pixels(line, grid)
        per_az = edges_d[:, rows, cols].mean(axis=1)
        az = float(_strike(*line))
        sym, step = cross_profile(dem_s, line, grid, cfg.profile_offset_m)
        records.append({
            "strike_deg": round(az, 1),
            "strike_class": strike_class(az),
            "length_m": round(float(L), 1),
            "length_in_aoi_m": round(geom.intersection(aoi_geom_utm).length, 1),
            "n_segments": int(n),
            "edge_coverage": round(float(union_d[rows, cols].mean()), 3),
            "support_azimuths": int((per_az >= cfg.support_min_fraction).sum()),
            "valley_support": round(float(rv_d[0][rows, cols].mean()), 3) if len(rv) else None,
            "ridge_support": round(float(rv_d[1][rows, cols].mean()), 3) if len(rv) else None,
            "profile_sym_m": round(sym, 2),
            "profile_step_m": round(step, 2),
            "morphology": profile_class(sym, step, cfg),
            "mean_slope_deg": round(float(np.degrees(slope[rows, cols]).mean()), 2),
        })
        geoms.append(geom)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=grid.crs)
    if len(gdf):
        gdf = gdf.sort_values("length_m", ascending=False).reset_index(drop=True)
        gdf.insert(0, "line_id", [f"L{i + 1:03d}" for i in range(len(gdf))])
    stats = {"hough_segments": n_raw, "after_merge": n_merged, "final_in_aoi": len(gdf)}
    return gdf, {"slope_deg": np.degrees(slope).astype("float32"),
                 "hillshade": hillshade(slope, aspect, 315.0, 45.0).astype("float32")}, stats


def lineament_rasters(gdf: gpd.GeoDataFrame, grid: Grid, radius_m: float) -> dict[str, np.ndarray]:
    """Distance to nearest lineament (m) and density (km of lineament per km^2) on ``grid``."""
    if len(gdf) == 0:
        return {"distance_m": np.full(grid.shape, np.nan, "float32"),
                "density_km_per_km2": np.zeros(grid.shape, "float32")}
    burn = rasterize([(g, 1) for g in gdf.geometry], out_shape=grid.shape, transform=grid.transform,
                     all_touched=True, dtype="uint8").astype(bool)
    distance = distance_transform_edt(~burn) * grid.res
    r = int(np.ceil(radius_m / grid.res))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = ((xx ** 2 + yy ** 2) * grid.res ** 2 <= radius_m ** 2).astype(float)
    # each burnt cell carries ~res metres of line; density = km / km^2
    length_km = fftconvolve(burn.astype(float) * grid.res / 1000.0, disk, mode="same")
    area_km2 = disk.sum() * (grid.res / 1000.0) ** 2
    return {"distance_m": distance.astype("float32"),
            "density_km_per_km2": np.clip(length_km / area_km2, 0, None).astype("float32")}


def rose_table(gdf: gpd.GeoDataFrame, bin_deg: float = 10.0):
    import pandas as pd
    edges = np.arange(0, 180 + bin_deg, bin_deg)
    if len(gdf) == 0:
        return pd.DataFrame({"bin_start_deg": edges[:-1], "bin_end_deg": edges[1:], "count": 0, "length_km": 0.0})
    idx = np.clip(np.digitize(gdf["strike_deg"], edges) - 1, 0, len(edges) - 2)
    count = np.bincount(idx, minlength=len(edges) - 1)
    length = np.bincount(idx, weights=gdf["length_in_aoi_m"] / 1000.0, minlength=len(edges) - 1)
    return pd.DataFrame({"bin_start_deg": edges[:-1], "bin_end_deg": edges[1:],
                         "count": count, "length_km": np.round(length, 3)})
