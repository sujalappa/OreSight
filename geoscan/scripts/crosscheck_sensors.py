"""Independent-sensor cross-check of the EMIT alteration layers.

    python scripts/crosscheck_sensors.py --name thanewasna

Do other sensors, processed separately from EMIT, show the same mineral patterns?

Independent indices (multi-scene median; per-scene vegetation mask NDVI <= 0.30, as for EMIT):
  ASTER L1T (2000-2007, SWIR working; DN -> radiance with the scene gain -> TOA reflectance)
      AlOH group content   (B5 + B7) / B6           CSIRO ASTER geoscience products (Cudahy 2012)
      MgOH group content   (B6 + B9) / (B7 + B8)    CSIRO
      Ferric oxide content  B4 / B3                  CSIRO
  Sentinel-2 L2A (2019-2024, Jan-May)
      Iron oxide ratio      B4 / B2                  red/blue ferric ratio (Sabins 1999, TM 3/1 equivalent)
      Ferric 900 nm         B11 / B8A                SWIR1/NIR, ASTER B4/B3 equivalent
      Clay / hydroxyl ratio B11 / B12                (Sabins 1999, TM 5/7 equivalent; Al-OH + Mg-OH + carbonate)
  Landsat 8/9 C2 L2 (2013-2024, Jan-May)
      Iron oxide ratio      SR_B4 / SR_B2
      Clay / hydroxyl ratio SR_B6 / SR_B7

Comparison with the published EMIT layer (intensity, 60 m), inside the AOI where EMIT data support >= 0.20:
  Spearman rho, with a p-value from random circular shifts (>= 1 km) of the independent map, which keeps
  its spatial autocorrelation (ordinary p-values would be far too optimistic);
  top-20 % overlap: share of the EMIT top-20 % pixels that are also top-20 % in the independent map (random = 20 %);
  and each independent index against ground truth (Thanewasna lodes / Kehalpur NGDR blocks).
Both maps are compared after the same 90 m Gaussian smoothing (the EMIT core kernel) and also unsmoothed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from geoscan.acquire import gee  # noqa: E402
from geoscan.aoi import load_aoi  # noqa: E402
from geoscan.grid import grid_for_aoi  # noqa: E402
from geoscan.raster_io import aoi_mask, read_raster, write_raster  # noqa: E402
from geoscan.validation import auc  # noqa: E402

ROOT = HERE.parents[1]
PROJECT = "animated-sign-467711-n9"
# ASTER exo-atmospheric irradiance (W m-2 um-1), Thome et al. (2001) values used by the ASTER science team
ESUN = {"B01": 1845.99, "B02": 1555.74, "B3N": 1119.47, "B04": 231.25, "B05": 79.81, "B06": 74.99,
        "B07": 68.66, "B08": 59.74, "B09": 56.92}
INDICES = {   # key: (sensor, label, compared EMIT layer)
    "aster_aloh": ("ASTER", "ASTER AlOH content (B5+B7)/B6", "muscovite"),
    "s2_clay": ("Sentinel-2", "Sentinel-2 clay ratio B11/B12", "muscovite"),
    "l8_clay": ("Landsat 8/9", "Landsat clay ratio B6/B7", "muscovite"),
    "aster_mgoh": ("ASTER", "ASTER MgOH content (B6+B9)/(B7+B8)", "chlorite"),
    "aster_ferric": ("ASTER", "ASTER ferric oxide content B4/B3", "gossan"),
    "s2_ferric900": ("Sentinel-2", "Sentinel-2 ferric B11/B8A", "gossan"),
    "s2_ironox": ("Sentinel-2", "Sentinel-2 iron oxide B4/B2", "gossan"),
    "l8_ironox": ("Landsat 8/9", "Landsat iron oxide B4/B2", "gossan"),
}


def ee_images(ee, region):
    def aster(img):
        doy = ee.Number(img.date().getRelative("day", "year")).add(1)
        d = ee.Number(1).subtract(ee.Number(0.01672).multiply(doy.subtract(4).multiply(0.9856 * np.pi / 180).cos()))
        sin_e = ee.Number(img.get("SOLAR_ELEVATION")).multiply(np.pi / 180).sin()
        refl = []
        for b, e in ESUN.items():
            gain = ee.Number(img.get(f"GAIN_COEFFICIENT_{b}"))
            dn = img.select(b)
            rad = dn.subtract(1).multiply(gain)
            refl.append(rad.multiply(np.pi).multiply(d.pow(2)).divide(sin_e.multiply(e)).updateMask(dn.gt(0)).rename(b))
        r = ee.Image(refl)
        ndvi = r.normalizedDifference(["B3N", "B02"])
        r = r.updateMask(ndvi.lte(0.30))
        return ee.Image([
            r.select("B05").add(r.select("B07")).divide(r.select("B06")).rename("aster_aloh"),
            r.select("B06").add(r.select("B09")).divide(r.select("B07").add(r.select("B08"))).rename("aster_mgoh"),
            r.select("B04").divide(r.select("B3N")).rename("aster_ferric")])

    ast = (ee.ImageCollection("ASTER/AST_L1T_003").filterBounds(region).filter(ee.Filter.lt("CLOUDCOVER", 10))
           .filter(ee.Filter.listContains("ORIGINAL_BANDS_PRESENT", "B09")).map(aster))

    def s2(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        r = img.select(["B2", "B4", "B8", "B8A", "B11", "B12"]).divide(10000).updateMask(ok)
        r = r.updateMask(r.normalizedDifference(["B8", "B4"]).lte(0.30))
        return ee.Image([r.select("B4").divide(r.select("B2")).rename("s2_ironox"),
                         r.select("B11").divide(r.select("B8A")).rename("s2_ferric900"),
                         r.select("B11").divide(r.select("B12")).rename("s2_clay")])

    s2c = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
           .filterDate("2019-01-01", "2025-01-01").filter(ee.Filter.calendarRange(1, 5, "month"))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)).map(s2))

    def l8(img):
        qa = img.select("QA_PIXEL")
        ok = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))   # cloud, cloud shadow
        r = img.select(["SR_B2", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]).multiply(2.75e-05).add(-0.2).updateMask(ok)
        r = r.updateMask(r.normalizedDifference(["SR_B5", "SR_B4"]).lte(0.30))
        return ee.Image([r.select("SR_B4").divide(r.select("SR_B2")).rename("l8_ironox"),
                         r.select("SR_B6").divide(r.select("SR_B7")).rename("l8_clay")])

    l8c = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(region).filterDate("2013-01-01", "2025-01-01")
           .filter(ee.Filter.calendarRange(1, 5, "month")).filter(ee.Filter.lt("CLOUD_COVER", 10)).map(l8))
    counts = {"ASTER": ast.size().getInfo(), "Sentinel-2": s2c.size().getInfo(), "Landsat 8/9": l8c.size().getInfo()}
    img = ee.Image([ast.median(), s2c.median(), l8c.median()])
    return img, counts


def to_grid(arr, src, dst, method=Resampling.average):
    out = np.full(dst.shape, np.nan, "float32")
    reproject(arr.astype("float32"), out, src_transform=src.transform, src_crs=src.crs, dst_transform=dst.transform,
              dst_crs=dst.crs, resampling=method, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def smooth(a, sigma_px):
    m = np.isfinite(a)
    num = gaussian_filter(np.where(m, a, 0), sigma_px)
    den = gaussian_filter(m.astype(float), sigma_px)
    return np.where(m, num / np.maximum(den, 1e-6), np.nan)


def shift_p(x, y, valid, res, n=300, min_m=1000.0, seed=0):
    """Spearman rho and a circular-shift p-value (one-sided, positive association)."""
    rho = spearmanr(x[valid], y[valid]).statistic
    rng = np.random.default_rng(seed)
    H, W = x.shape
    k = int(min_m / res)
    null = []
    while len(null) < n:
        dr, dc = rng.integers(0, H), rng.integers(0, W)
        if min(dr, H - dr) < k and min(dc, W - dc) < k:
            continue
        ys = np.roll(np.roll(y, dr, 0), dc, 1)
        v = valid & np.isfinite(ys)
        if v.sum() > 50:
            null.append(spearmanr(x[v], ys[v]).statistic)
    null = np.array(null)
    return float(rho), float((1 + (null >= rho).sum()) / (1 + len(null))), float(np.percentile(null, 95))


def top_overlap(x, y, valid, q=0.80):
    tx = x >= np.quantile(x[valid], q)
    ty = y >= np.quantile(y[valid], q)
    return float((tx & ty & valid).sum() / max((tx & valid).sum(), 1))


def truth(name, grid, aoi_utm, inside):
    gt = ROOT / "data" / name / "ground_truth"
    if name == "thanewasna":
        lodes = gpd.read_file(gt / "cu_lodes_mineral_zone.geojson").to_crs(grid.crs)
        return {"lodes": rasterize([(g, 1) for g in lodes.geometry], out_shape=grid.shape, transform=grid.transform,
                                   all_touched=True).astype(bool) & inside}
    idx = gpd.read_file(gt / "ngdr_report_index/EXPLORATION_DATA_GIS_VIEW_REPORT_CLIP.shp").to_crs(grid.crs)
    out = {}
    for rid in ("CRO-20859-1986", "CRO-23431-2018", "CRO-22666-2009"):
        poly = idx[idx.exp_upid == rid].union_all().intersection(aoi_utm)
        out[rid] = rasterize([(poly, 1)], out_shape=grid.shape, transform=grid.transform).astype(bool) & inside
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    out = ROOT / "outputs" / f"{a.name}_sensor_crosscheck"
    for d in ("tables", "maps", "rasters"):
        (out / d).mkdir(parents=True, exist_ok=True)
    aoi = load_aoi(ROOT / "outputs" / a.name / "vectors" / "aoi.geojson", a.name)
    g30 = grid_for_aoi(aoi, 30.0, 180.0)
    aoi_utm = aoi.to_crs(g30.crs)

    raw = ROOT / "data" / a.name / "raw" / "independent_sensors" / "indices_30m.tif"
    if not raw.exists():
        gee.initialize(PROJECT)
        ee = gee._ee()
        region = ee.Geometry.Rectangle(list(aoi.bounds)).buffer(1000)
        img, counts = ee_images(ee, region)
        print("scenes used:", counts, flush=True)
        arr = gee.fetch(img, list(INDICES), g30)
        write_raster(raw, arr, g30, list(INDICES), tags={"scenes": str(counts)})
    ind, g30, names, tags = read_raster(raw)
    ind = {n: ind[i] for i, n in enumerate(names)}
    write_raster(out / "rasters" / "independent_indices_30m.tif", np.stack([ind[n] for n in INDICES]), g30,
                 list(INDICES), tags=tags)

    emit = {}
    for m in ("muscovite", "chlorite", "gossan"):
        d, ge, bn, _ = read_raster(ROOT / "outputs" / a.name / "rasters" / "alteration" / f"{m}.tif")
        emit[m] = d[bn.index("intensity")]
        support = d[bn.index("data_support")]
    inside_e = aoi_mask(ge, aoi_utm)
    valid_e = inside_e & (np.nan_to_num(support) >= 0.20)
    inside30 = aoi_mask(g30, aoi_utm)
    gt30 = truth(a.name, g30, aoi_utm, inside30)

    rows, gt_rows = [], []
    for key, (sensor, label, m) in INDICES.items():
        y = to_grid(ind[key], g30, ge)
        for sm, sig in (("smoothed 90 m", 90.0 / ge.res), ("unsmoothed", 0.0)):
            xs = smooth(emit[m], sig) if sig else emit[m]
            ys = smooth(y, sig) if sig else y
            v = valid_e & np.isfinite(xs) & np.isfinite(ys)
            rho, p, null95 = shift_p(xs, ys, v, ge.res)
            rows.append({"emit_layer": m, "independent_index": label, "sensor": sensor, "comparison": sm,
                         "pixels": int(v.sum()), "spearman_rho": round(rho, 3), "shift_null_95pct": round(null95, 3),
                         "p_shift": round(p, 4), "top20_overlap_pct": round(100 * top_overlap(xs, ys, v), 1),
                         "top20_overlap_random_pct": 20.0})
        v30 = inside30 & np.isfinite(ind[key])
        for gk, mask in gt30.items():
            gt_rows.append({"index": label, "ground_truth": gk, "auc": round(auc(ind[key][v30 & mask], ind[key][v30 & ~mask]), 3)})
    # EMIT layers against the same ground truth, for reference
    for m in ("muscovite", "chlorite", "gossan"):
        e30 = to_grid(emit[m], ge, g30, Resampling.bilinear)
        v30 = inside30 & np.isfinite(e30)
        for gk, mask in gt30.items():
            gt_rows.append({"index": f"EMIT {m} (published layer)", "ground_truth": gk,
                            "auc": round(auc(e30[v30 & mask], e30[v30 & ~mask]), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "tables" / "agreement_with_emit.csv", index=False)
    gdf = pd.DataFrame(gt_rows).pivot(index="index", columns="ground_truth", values="auc")
    gdf.to_csv(out / "tables" / "ground_truth_auc.csv")
    pd.set_option("display.width", 250)
    print(df.to_string())
    print(gdf.to_string())
    panels(out, a.name, emit, ind, ge, g30, valid_e, aoi_utm)


def panels(out, name, emit, ind, ge, g30, valid_e, aoi_utm):
    ext_e = [ge.x0, ge.x0 + ge.width * ge.res, ge.y0 - ge.height * ge.res, ge.y0]
    ext_30 = [g30.x0, g30.x0 + g30.width * g30.res, g30.y0 - g30.height * g30.res, g30.y0]
    inside30 = aoi_mask(g30, aoi_utm)
    for m in ("muscovite", "chlorite", "gossan"):
        keys = [k for k, v in INDICES.items() if v[2] == m]
        fig, axs = plt.subplots(1, 1 + len(keys), figsize=(5 * (1 + len(keys)), 5.5), dpi=120)
        axs = np.atleast_1d(axs)
        layers = [(f"EMIT {m} (published)", np.where(valid_e, emit[m], np.nan), ext_e)] + \
                 [(INDICES[k][1], np.where(inside30, smooth(ind[k], 90.0 / g30.res), np.nan), ext_30) for k in keys]
        for ax, (t, arr, ext) in zip(axs, layers):
            lo, hi = np.nanpercentile(arr, [5, 98])
            ax.imshow(arr, extent=ext, cmap="inferno", vmin=lo, vmax=hi)
            gpd.GeoSeries([aoi_utm]).boundary.plot(ax=ax, color="cyan", lw=1)
            ax.set_title(t, fontsize=9, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{name}: {m}. Bright = stronger signal (each map stretched P5-P98); "
                     f"independent maps smoothed 90 m", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "maps" / f"crosscheck_{m}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
