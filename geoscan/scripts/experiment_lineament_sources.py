"""Experiment: which data source / window size recovers the ore-controlling structure?

    python scripts/experiment_lineament_sources.py --name thanewasna [--pad-km 8]

Same edge -> Hough -> merge -> length-filter chain as the pipeline, run on:
  DEMs (hillshade + Sato):  SRTM (current), Copernicus GLO-30, AW3D30 v4.1
  Images (CLAHE + Canny + Sato on the image itself): Sentinel-1 VV / VH dry-season median,
  Sentinel-2 dry-season median PC1
over a regional window (AOI padded by --pad-km) instead of the current 1.5 km pad.

Scores (Thanewasna):
  lode AUC / borehole AUC of proximity to the nearest lineament, inside the AOI
  share of lineament length within 3 km of the AOI striking 155 +/- 15 deg (lode / GSI shear strike)
  GSI shear recovery: length of lineaments within 200 m and 15 deg of the mapped shear (extended 2 km)
Outputs: outputs/thanewasna_lineament_sources/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
from shapely.affinity import scale as shp_scale
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from skimage.filters import sato
from skimage.morphology import skeletonize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from geoscan.acquire import gee  # noqa: E402
from geoscan.aoi import load_aoi  # noqa: E402
from geoscan.config import load_settings  # noqa: E402
from geoscan.grid import Grid, grid_for_aoi  # noqa: E402
from geoscan.raster_io import read_raster, write_raster  # noqa: E402
from geoscan.structure import lineaments as lin  # noqa: E402
from geoscan.validation import auc, bootstrap_auc  # noqa: E402

ROOT = HERE.parents[1]
DRY = ("2019-01-01", "2024-12-31", 1, 5)          # years, months Jan-May (dry, leaf-off)


def _grid(aoi, res, pad_m):
    g = grid_for_aoi(aoi, res, 180.0)
    return g.pad(pad_m)


def sources(ee, region):
    def dem(coll, band):
        c = ee.ImageCollection(coll).filterBounds(region).select(band)
        return c.mosaic().setDefaultProjection(c.first().projection()).resample("bilinear")

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filterDate(DRY[0], DRY[1]).filter(ee.Filter.calendarRange(DRY[2], DRY[3], "month")))
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
          .filterDate(DRY[0], DRY[1]).filter(ee.Filter.calendarRange(DRY[2], DRY[3], "month"))
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)))
    return {
        "dem_srtm": (ee.Image("USGS/SRTMGL1_003").select("elevation").resample("bilinear"), ["elevation"], 30),
        "dem_copernicus": (dem("COPERNICUS/DEM/GLO30", "DEM"), ["DEM"], 30),
        "dem_aw3d30": (dem("JAXA/ALOS/AW3D30/V4_1", "DSM"), ["DSM"], 30),
        "s1_vv_vh": (s1.select(["VV", "VH"]).median(), ["VV", "VH"], 10),
        "s2_dry": (s2.select(["B2", "B3", "B4", "B8", "B11", "B12"]).median(),
                   ["B2", "B3", "B4", "B8", "B11", "B12"], 10),
    }


def image_lineaments(img: np.ndarray, grid: Grid, aoi_geom, cfg):
    """Pipeline chain on an image: CLAHE + Canny and Sato bright/dark axes -> Hough -> merge."""
    img = np.where(np.isfinite(img), img, np.nanmedian(img))
    lo, hi = np.percentile(img, [1, 99])
    x = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)
    x = gaussian_filter(x, 15.0 / grid.res)              # ~15 m smoothing (speckle / texture)
    kernel = max(8, int(round(cfg.clahe_window_m / grid.res)))
    xe = equalize_adapthist(x, kernel_size=kernel, clip_limit=cfg.clahe_clip)
    edges = tiled_canny(xe, grid.res, 30.0 / grid.res, cfg) if CANNY == "tiled" else \
        canny(xe, sigma=30.0 / grid.res, low_threshold=cfg.canny_low_quantile,
              high_threshold=cfg.canny_high_quantile, use_quantiles=True)
    sig = [s / grid.res for s in cfg.ridge_sigmas_m]
    win = cfg.clahe_window_m / grid.res / 2
    axes = []
    for black in (True, False):
        r = sato(x, sigmas=sig, black_ridges=black, mode="reflect")
        r = r / (gaussian_filter(r, win) + 1e-9)
        axes.append(skeletonize(r > np.quantile(r, cfg.ridge_quantile)))
    union = edges | axes[0] | axes[1]
    return _hough(union, grid, aoi_geom, cfg)


def _hough(union, grid, aoi_geom, cfg):
    lines = cv2.HoughLinesP(union.astype(np.uint8) * 255, 1, np.pi / 180, cfg.hough_threshold,
                            minLineLength=int(round(cfg.min_segment_m / grid.res)),
                            maxLineGap=int(round(cfg.max_gap_m / grid.res)))
    raw = np.zeros((0, 4)) if lines is None else lines[:, 0, :].astype(float)
    seg = np.column_stack([grid.x0 + (raw[:, 0] + 0.5) * grid.res, grid.y0 - (raw[:, 1] + 0.5) * grid.res,
                           grid.x0 + (raw[:, 2] + 0.5) * grid.res, grid.y0 - (raw[:, 3] + 0.5) * grid.res])
    seg = seg[np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= cfg.min_segment_m]
    seg, count = lin.merge_collinear(seg, cfg.merge_angle_deg, cfg.merge_perp_m, cfg.merge_gap_m)
    seg, count = lin.remove_duplicates(seg, count, cfg.dedup_angle_deg, cfg.dedup_perp_m)
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    seg = seg[L >= cfg.min_length_m]
    from shapely.geometry import LineString
    g = gpd.GeoDataFrame({"strike_deg": lin._strike(*seg.T) if len(seg) else []},
                         geometry=[LineString([s[:2], s[2:]]) for s in seg], crs=grid.crs)
    g["length_m"] = g.length
    return g[g.intersects(aoi_geom.buffer(3000))].reset_index(drop=True)


def score(g, ref, aoi_geom, lodes, bh, shear):
    d = _distance(g, ref)
    inside = rasterize([(aoi_geom, 1)], out_shape=ref.shape, transform=ref.transform).astype(bool)
    lode = rasterize([(x, 1) for x in lodes.geometry], out_shape=ref.shape, transform=ref.transform,
                     all_touched=True).astype(bool) & inside
    br = ((ref.y0 - bh.y.values) / ref.res).astype(int)
    bc = ((bh.x.values - ref.x0) / ref.res).astype(int)
    b = bootstrap_auc(-d[br, bc][bh.lode.values], -d[br, bc][~bh.lode.values], n=1000)
    near = g[g.distance(aoi_geom) <= 3000]
    w = near.geometry.intersection(aoi_geom.buffer(3000)).length
    para = lin._angle_diff(near.strike_deg.values, 155.5) <= 15
    sh_ext = shp_scale(shear, xfact=1 + 4000 / shear.length, yfact=1 + 4000 / shear.length)
    sb = sh_ext.buffer(200)
    hit = near[(lin._angle_diff(near.strike_deg.values, 155.5) <= 15) & near.intersects(sb)]
    return {"n_lines_3km": len(near), "km_lines_3km": round(w.sum() / 1000, 2),
            "pct_length_nw_155": round(100 * w[para].sum() / max(w.sum(), 1e-9), 1),
            "shear_recovered_m": round(hit.geometry.intersection(sb).length.sum()),
            "lode_auc": round(auc(-d[lode], -d[inside & ~lode]), 3),
            "borehole_auc": round(b["auc"], 3), "borehole_p": round(b["p_value"], 3),
            "median_dist_lode_m": round(float(np.median(d[lode]))),
            "median_dist_block_m": round(float(np.median(d[inside])))}


TILE_M = 3000.0
CANNY = "global"


def tiled_canny(img, res, sigma, cfg, tile_m=TILE_M):
    """Canny with quantile thresholds computed per ~3 km tile (+25% overlap), so flat ground is
    not out-competed by high-relief ground elsewhere in a large window."""
    T = max(32, int(tile_m / res))
    P = T // 4
    out = np.zeros(img.shape, bool)
    for r0 in range(0, img.shape[0], T):
        for c0 in range(0, img.shape[1], T):
            ra, ca = max(0, r0 - P), max(0, c0 - P)
            sub = img[ra:r0 + T + P, ca:c0 + T + P]
            e = canny(sub, sigma=sigma, low_threshold=cfg.canny_low_quantile,
                      high_threshold=cfg.canny_high_quantile, use_quantiles=True)
            out[r0:r0 + T, c0:c0 + T] = e[r0 - ra:r0 - ra + T, c0 - ca:c0 - ca + T]
    return out


_orig_hillshade_edges = lin.hillshade_edges


def tiled_hillshade_edges(slope, aspect, grid, cfg):
    kernel = max(8, int(round(cfg.clahe_window_m / grid.res)))
    out = []
    for az in np.arange(cfg.n_azimuths) * (360.0 / cfg.n_azimuths):
        hs = lin.hillshade(slope, aspect, az, cfg.sun_elevation_deg)
        hs = equalize_adapthist(hs, kernel_size=kernel, clip_limit=cfg.clahe_clip)
        out.append(tiled_canny(hs, grid.res, 1.0, cfg))
    return np.stack(out)


def set_mode(mode):
    global CANNY
    lin.hillshade_edges = tiled_hillshade_edges if mode == "tiled" else _orig_hillshade_edges
    CANNY = mode


def segments(geoms):
    out = []
    for g in geoms:
        for part in getattr(g, "geoms", [g]):
            c = np.asarray(part.coords)[:, :2]
            out += [np.r_[c[k], c[k + 1]] for k in range(len(c) - 1)]
    return np.array(out).reshape(-1, 4)


def _distance(g, ref):
    from scipy.ndimage import distance_transform_edt
    # burn on a grid padded by 3 km so lines that pass near (not through) the AOI still count
    big = ref.pad(3000.0)
    burn = rasterize([(x, 1) for x in g.geometry], out_shape=big.shape, transform=big.transform,
                     all_touched=True, dtype="uint8").astype(bool) if len(g) else np.zeros(big.shape, bool)
    if not burn.any():
        return np.full(ref.shape, 1e4)
    rs, cs = ref.window_in(big)
    return (distance_transform_edt(~burn) * ref.res)[rs, cs]


def score_kehalpur(g, ref, aoi_geom, ctx):
    """No lodes at Kehalpur: agreement with GSI-mapped lineaments and foliation, and NGDR blocks."""
    from shapely.geometry import LineString
    gi = g[g.intersects(aoi_geom)].copy()
    w = gi.geometry.intersection(aoi_geom).length.values
    az = gi.strike_deg.values.astype(float)
    fol = ctx["fol"]
    fd, fa = [], []
    for m in gi.geometry.interpolate(0.5, normalized=True):
        dd = fol.distance(m)
        k = dd.idxmin()
        fd.append(dd[k])
        fa.append(fol.loc[k, "axial"])
    fd, fa = np.array(fd), np.array(fa)
    near_f = fd <= 2000
    fol_ok = near_f & (lin._angle_diff(az, fa) <= 20) if len(gi) else np.zeros(0, bool)
    rec_len = tot = 0.0
    for s in ctx["gsi_seg"]:
        ls = LineString([s[:2], s[2:]]).intersection(aoi_geom)
        if ls.is_empty:
            continue
        tot += ls.length
        cand = gi[lin._angle_diff(az, float(lin._strike(*s))) <= 15]
        if len(cand):
            rec_len += ls.intersection(cand.geometry.buffer(300).union_all()).length
    d = _distance(g, ref)
    inside, blk, base = ctx["inside"], ctx["blk"], ctx["basement"]
    return {"n_lines_aoi": len(gi), "km_lines_aoi": round(w.sum() / 1000, 1),
            "pct_len_foliation_63": round(100 * w[lin._angle_diff(az, 63.0) <= 20].sum() / max(w.sum(), 1e-9), 1),
            "pct_len_parallel_local_foliation": round(100 * w[fol_ok].sum() / max(w[near_f].sum(), 1e-9), 1),
            "gsi_lineament_recall_pct": round(100 * rec_len / max(tot, 1e-9), 1),
            "ngdr_block_auc": round(auc(-d[blk & inside], -d[inside & ~blk]), 3),
            "ngdr_block_auc_basement": round(auc(-d[blk & inside & base], -d[inside & base & ~blk]), 3)}


def score_thanewasna(g, ref, aoi_geom, ctx):
    return score(g, ref, aoi_geom, ctx["lodes"], ctx["bh"], ctx["shear"])


def context(name, aoi, aoi_geom, ref):
    gt = ROOT / "data" / name / "ground_truth"
    if name == "thanewasna":
        lodes = gpd.read_file(gt / "cu_lodes_mineral_zone.geojson").to_crs(aoi.utm_crs)
        shear = gpd.read_file(gt / "bhukosh/LINEAMENT_250K_CLIP").to_crs(aoi.utm_crs).geometry.iloc[0]
        bh = pd.read_csv(gt / "Thanewasna_Borehole_Ground_Truth_Master.csv")
        bh["lode"] = bh["Intersected_Cu_Lode"].str.lower().str.startswith("lode")
        p = gpd.GeoSeries(gpd.points_from_xy(bh.BH_Lon, bh.BH_Lat), crs=4326).to_crs(aoi.utm_crs)
        bh["x"], bh["y"] = p.x.values, p.y.values
        return {"lodes": lodes, "shear": shear, "bh": bh}, score_thanewasna

    def burn(geoms):
        geoms = list(geoms)
        return rasterize([(x, 1) for x in geoms], out_shape=ref.shape, transform=ref.transform).astype(bool) \
            if geoms else np.zeros(ref.shape, bool)

    fol = gpd.read_file(gt / "structure_planes_gsi_cro19966.geojson").to_crs(aoi.utm_crs)
    fol["axial"] = fol["STRIKE"].astype(float) % 180
    gsi = gpd.read_file(gt / "bhukosh/LINEAMENT_250K_CLIP").to_crs(aoi.utm_crs)
    idx = gpd.read_file(gt / "ngdr_report_index/EXPLORATION_DATA_GIS_VIEW_REPORT_CLIP.shp").to_crs(aoi.utm_crs)
    blk = idx[idx.exp_upid.isin(["CRO-20859-1986", "CRO-23431-2018", "CRO-22666-2009"])]
    lit = gpd.read_file(gt / "lithology_gsi_cro19966.geojson").to_crs(aoi.utm_crs)
    cover = lit[lit.LITHOLOGICAL_UNIT.str.contains("Deccan|Infra", na=False)]
    return {"fol": fol, "gsi_seg": segments(gsi.geometry), "inside": burn([aoi_geom]),
            "blk": burn(blk.geometry), "basement": ~burn(cover.geometry)}, score_kehalpur


def _references(name, aoi, scorer, ref, aoi_geom, ctx):
    cur = gpd.read_file(ROOT / "outputs" / name / "vectors/lineaments.geojson").to_crs(aoi.utm_crs)
    cur = cur[cur.morphology != "scarp"]
    cur["strike_deg"] = cur["strike_deg"].astype(float)
    gsi = gpd.read_file(ROOT / "data" / name / "ground_truth/bhukosh/LINEAMENT_250K_CLIP").to_crs(aoi.utm_crs)
    s = segments(gsi.geometry)
    from shapely.geometry import LineString
    gsi = gpd.GeoDataFrame({"strike_deg": lin._strike(*s.T)},
                           geometry=[LineString([r[:2], r[2:]]) for r in s], crs=aoi.utm_crs)
    return [{"source": "current_pipeline", "res_m": 30, "pad_km": 1.5, "thresholds": "global",
             **scorer(cur, ref, aoi_geom, ctx)},
            {"source": "GSI 1:250k lineaments (reference)", **scorer(gsi, ref, aoi_geom, ctx)}]


def rescore(a, cfg):
    out = ROOT / "outputs" / f"{a.name}_lineament_sources"
    aoi = load_aoi(ROOT / "data" / a.name / "aoi" / "aoi.geojson", a.name)
    aoi_geom = aoi.to_crs(aoi.utm_crs)
    ref = grid_for_aoi(aoi, 30.0, 180.0)
    ctx, scorer = context(a.name, aoi, aoi_geom, ref)
    runs = []
    for f in sorted((out / "vectors").glob("*_pad*km_*.geojson")):
        stem = f.stem
        name, rest = stem.split("_pad")
        pad, mode = rest.split("km_")
        g = gpd.read_file(f)
        if len(g) == 0:
            g = gpd.GeoDataFrame({"strike_deg": []}, geometry=[], crs=aoi.utm_crs)
        g = g.to_crs(aoi.utm_crs)
        if "morphology" in g and cfg.exclude_scarps:
            g = g[g.morphology != "scarp"]
        g["strike_deg"] = g["strike_deg"].astype(float)
        runs.append({"source": name, "res_m": 10 if name.startswith("s") else 30, "pad_km": float(pad),
                     "thresholds": mode, **scorer(g, ref, aoi_geom, ctx)})
    runs += _references(a.name, aoi, scorer, ref, aoi_geom, ctx)
    df = pd.DataFrame(runs)
    df.to_csv(out / "lineament_source_scores.csv", index=False)
    print(df.to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="thanewasna")
    ap.add_argument("--pad-km", type=float, default=8.0)
    ap.add_argument("--rescore", action="store_true", help="re-score saved vectors only")
    a = ap.parse_args(argv)
    st = load_settings(None)
    cfg = st.structure
    data = ROOT / "data" / a.name
    if a.rescore:
        return rescore(a, cfg)
    raw = data / "raw" / "lineament_sources"
    raw.mkdir(parents=True, exist_ok=True)
    out = ROOT / "outputs" / f"{a.name}_lineament_sources"
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    aoi = load_aoi(data / "aoi" / "aoi.geojson", a.name)
    aoi_geom = aoi.to_crs(aoi.utm_crs)

    gee.initialize(st.acquire.ee_project or "animated-sign-467711-n9")
    ee = gee._ee()
    region = ee.Geometry.Rectangle(list(aoi.bounds)).buffer(a.pad_km * 1000 + 2000)
    src = sources(ee, region)
    ref = grid_for_aoi(aoi, 30.0, 180.0)
    ctx, scorer = context(a.name, aoi, aoi_geom, ref)

    runs = []
    for key, (img, bands, res) in src.items():
        g = _grid(aoi, res, a.pad_km * 1000)
        f = raw / f"{key}_{int(a.pad_km)}km_{res}m.tif"
        if not f.exists():
            print("download", key, g.shape, flush=True)
            arr = gee.fetch(img, bands, g, st.acquire.max_request_mb, st.acquire.retries)
            write_raster(f, arr, g, bands, tags={"source": key})
        arr, g, names, _ = read_raster(f)
        if key.startswith("dem"):
            variants = {key: arr[0]}
        elif key == "s1_vv_vh":
            variants = {"s1_vv": arr[0], "s1_vh": arr[1]}
        else:
            X = arr.reshape(len(arr), -1).T
            ok = np.isfinite(X).all(1)
            Z = (X[ok] - X[ok].mean(0)) / X[ok].std(0)
            pc1 = np.full(X.shape[0], np.nan)
            pc1[ok] = Z @ np.linalg.eigh(np.cov(Z.T))[1][:, -1]
            variants = {"s2_pc1": pc1.reshape(arr.shape[1:])}
        small = _grid(aoi, res, 1500.0)
        rs, cs = small.window_in(g)
        for name, x in variants.items():
            for pad, mode in ((1.5, "global"), (1.5, "tiled"), (a.pad_km, "tiled")):
                gg, xx = (small, x[rs, cs]) if pad == 1.5 else (g, x)
                set_mode(mode)
                if key.startswith("dem"):
                    gdf_all, _, _ = lin.extract_lineaments(xx, gg, aoi_geom, replace(cfg))
                    gdf_all = gdf_all[gdf_all.intersects(aoi_geom.buffer(3000))]
                    gdf = gdf_all[gdf_all.morphology != "scarp"] if cfg.exclude_scarps else gdf_all
                else:
                    gdf = gdf_all = image_lineaments(xx, gg, aoi_geom, cfg)
                tag = f"{name}_pad{pad:g}km_{mode}"
                gdf_all.to_file(out / "vectors" / f"{tag}.geojson", driver="GeoJSON")
                r = {"source": name, "res_m": res, "pad_km": pad, "thresholds": mode,
                     **scorer(gdf, ref, aoi_geom, ctx)}
                print(r, flush=True)
                runs.append(r)
    cur = gpd.read_file(ROOT / "outputs" / a.name / "vectors/lineaments.geojson").to_crs(aoi.utm_crs)
    cur = cur[cur.morphology != "scarp"]
    cur["strike_deg"] = cur["strike_deg"].astype(float)
    runs.append({"source": "current_pipeline", "res_m": 30, "pad_km": 1.5, "thresholds": "global",
                 **scorer(cur, ref, aoi_geom, ctx)})
    gsi = gpd.read_file(ROOT / "data" / a.name / "ground_truth/bhukosh/LINEAMENT_250K_CLIP").to_crs(aoi.utm_crs)
    s = segments(gsi.geometry)
    gsi = gpd.GeoDataFrame({"strike_deg": lin._strike(*s.T)},
                           geometry=[__import__("shapely").geometry.LineString([r[:2], r[2:]]) for r in s],
                           crs=aoi.utm_crs)
    runs.append({"source": "GSI 1:250k lineaments (reference)", **scorer(gsi, ref, aoi_geom, ctx)})
    df = pd.DataFrame(runs)
    df.to_csv(out / "lineament_source_scores.csv", index=False)
    print(df.to_string())


def plot(out, aoi_geom, lodes, shear, names):
    n = len(names)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 10), dpi=110)
    view = aoi_geom.buffer(3000).bounds
    for ax, name in zip(axs.ravel(), names):
        g = gpd.read_file(out / "vectors" / f"{name}.geojson")
        para = lin._angle_diff(g.strike_deg.values, 155.5) <= 15
        g[~para].plot(ax=ax, color="#9e9e9e", lw=0.8)
        g[para].plot(ax=ax, color="#d50000", lw=1.6)
        gpd.GeoSeries([aoi_geom]).boundary.plot(ax=ax, color="black", lw=1.2)
        lodes.plot(ax=ax, color="#00c853")
        gpd.GeoSeries([shear]).plot(ax=ax, color="#2962ff", lw=3, alpha=0.6)
        ax.set_xlim(view[0], view[2]); ax.set_ylim(view[1], view[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=10, fontweight="bold")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Lineaments within 3 km of Thanewasna (red = 155±15° like the lodes, grey = other; "
                 "blue = GSI shear zone, green = Cu lodes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "lineament_sources.png")


if __name__ == "__main__":
    main()
