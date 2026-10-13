"""Experiment: does adding EMIT L2B evidence (weight = fit / uncertainty) improve the alteration layers?

    python scripts/experiment_l2b.py

Runs the geoscan ``alteration`` and ``targets`` steps for Thanewasna and Kehalpur with
alteration.l2b_evidence in {none, combine_all, combine_valid} and scores each variant with the
same tests used for the method comparison:

    Thanewasna  CPI vs 30 Cu lode polygons (pixel AUC, lode area in top 20 %), CPI at 47 borehole
                collars (8 lode vs 39 halo), top-10 targets within 100 m of a lode, and the AUC of
                the muscovite / chlorite / gossan layers themselves.
    Kehalpur     geological consistency (basement vs Deccan cover; host vs non-host lithology)
                and CPI inside NGDR exploration-block footprints.

Writes outputs/l2b_experiment/results.csv. Leaves both blocks re-run with the baseline ("none").
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from compare_lithology import COVER, HOST, NONHOST, lith_group  # noqa: E402
from compare_methods import geoscan_layers  # noqa: E402
from geoscan.aoi import load_aoi  # noqa: E402
from geoscan.config import load_settings  # noqa: E402
from geoscan.grid import grid_for_aoi  # noqa: E402
from geoscan.pipeline import Pipeline  # noqa: E402
from geoscan.raster_io import aoi_mask  # noqa: E402
from geoscan.validation import bootstrap_auc, captured_at, success_rate, target_hits  # noqa: E402

ROOT = HERE.parents[1]
CONFIG = HERE.parent / "config" / "default.yaml"
AOIS = {"thanewasna": ROOT / "data/thanewasna/aoi/thanewasna_emit_download_area.geojson",
        "kehalpur": ROOT / "data/kehalpur/aoi/kehalpur_Cu_Known_Block.geojson"}
VARIANTS = ("none", "combine_all", "combine_valid")
rng = np.random.default_rng(0)


def auc_of(arr, pos_mask, neg_mask, valid, cap=3000, n=300):
    v = valid & np.isfinite(arr)
    pos, neg = arr[v & pos_mask], arr[v & neg_mask]
    pos = rng.choice(pos, min(cap, pos.size), replace=False) if pos.size > cap else pos
    neg = rng.choice(neg, min(cap, neg.size), replace=False) if neg.size > cap else neg
    return bootstrap_auc(pos, neg, n=n)


def score_thanewasna():
    name = "thanewasna"
    aoi = load_aoi(AOIS[name], name)
    grid = grid_for_aoi(aoi, 15.0, 180.0)
    aoi_utm = aoi.to_crs(grid.crs)
    inside = aoi_mask(grid, aoi_utm)
    G = geoscan_layers(ROOT / "outputs" / name, grid, inside)
    lodes = gpd.read_file(ROOT / "data/thanewasna/ground_truth/cu_lodes_mineral_zone.geojson").to_crs(grid.crs)
    lode = rasterize([(g, 1) for g in lodes.geometry], out_shape=grid.shape, transform=grid.transform,
                     all_touched=True, dtype="uint8").astype(bool) & inside
    bh = pd.read_csv(ROOT / "data/thanewasna/ground_truth/Thanewasna_Borehole_Ground_Truth_Master.csv")
    pos = bh["Intersected_Cu_Lode"].str.lower().str.startswith("lode").values
    bx, by = Transformer.from_crs(4326, grid.crs, always_xy=True).transform(bh.BH_Lon.values, bh.BH_Lat.values)
    bc = np.clip(((bx - grid.x0) / grid.res).astype(int), 0, grid.width - 1)
    br = np.clip(((grid.y0 - by) / grid.res).astype(int), 0, grid.height - 1)
    r = {}
    for layer in ("cpi", "muscovite", "chlorite", "gossan"):
        a = G[layer]
        lo = auc_of(a, lode, ~lode, inside)
        b = bootstrap_auc(a[br, bc][pos], a[br, bc][~pos], n=500)
        r[f"TW_{layer}_lode_auc"] = lo["auc"]
        r[f"TW_{layer}_bh_auc"] = b["auc"]
        if layer == "cpi":
            r["TW_cpi_bh_p"] = b["p_value"]
            r["TW_cpi_lode_top20"] = captured_at(success_rate(a, lode, inside & np.isfinite(a)))["captured_top20pct"]
    t = pd.read_csv(ROOT / "outputs" / name / "tables/top50_targets.csv")
    h = target_hits(t[["Easting", "Northing"]].to_numpy()[:10], lodes.geometry, aoi_utm)
    r["TW_top10_hits100m"] = h.get("hits_within_100m")
    return r


def score_kehalpur():
    name = "kehalpur"
    aoi = load_aoi(AOIS[name], name)
    grid = grid_for_aoi(aoi, 15.0, 180.0)
    aoi_utm = aoi.to_crs(grid.crs)
    inside = aoi_mask(grid, aoi_utm)
    G = geoscan_layers(ROOT / "outputs" / name, grid, inside)
    lith = gpd.read_file(ROOT / "data/kehalpur/ground_truth/lithology_gsi_cro19966.geojson").to_crs(grid.crs)
    lith["group"] = lith["LITHOLOGICAL_UNIT"].map(lith_group)
    cls = np.zeros(grid.shape, "uint8")
    for g in (COVER, NONHOST, HOST):
        sub = lith[lith["group"] == g]
        cls[rasterize([(x, 1) for x in sub.geometry], out_shape=grid.shape, transform=grid.transform,
                      dtype="uint8").astype(bool)] = g
    cls[~inside] = 0
    c = G["cpi"]
    r = {"KP_A_basement_vs_cover": auc_of(c, (cls == HOST) | (cls == NONHOST), cls == COVER, inside)["auc"],
         "KP_B_host_vs_nonhost": auc_of(c, cls == HOST, cls == NONHOST, inside)["auc"]}
    t = pd.read_csv(ROOT / "outputs" / name / "tables/top50_targets.csv")
    tc = np.clip(((t.Easting.values - grid.x0) / grid.res).astype(int), 0, grid.width - 1)
    tr = np.clip(((grid.y0 - t.Northing.values) / grid.res).astype(int), 0, grid.height - 1)
    r["KP_targets_on_host"] = int((cls[tr, tc] == HOST).sum())
    idx = gpd.read_file(ROOT / "data/kehalpur/ground_truth/ngdr_report_index/EXPLORATION_DATA_GIS_VIEW_REPORT_CLIP.shp"
                        ).to_crs(grid.crs)
    for rid in ("CRO-20859-1986", "CRO-23431-2018", "CRO-22666-2009", "CRO-21815-1993"):
        poly = idx[idx.exp_upid == rid].union_all().intersection(aoi_utm)
        m = rasterize([(poly, 1)], out_shape=grid.shape, transform=grid.transform).astype(bool) & inside
        r[f"KP_{rid}_auc"] = auc_of(c, m, ~m, inside)["auc"]
        r[f"KP_{rid}_targets"] = int(sum(poly.contains(Point(x, y)) for x, y in t[["Easting", "Northing"]].to_numpy()))
    return r


def run(name, variant):
    s = load_settings(CONFIG, {"alteration": {"l2b_evidence": variant}})
    Pipeline(AOIS[name], name, ROOT, s).run(("alteration", "targets"), clean=False)


def main():
    out = ROOT / "outputs/l2b_experiment"
    out.mkdir(parents=True, exist_ok=True)
    for name in AOIS:  # refresh L2B rasters (adds weight / weighted_depth)
        Pipeline(AOIS[name], name, ROOT, load_settings(CONFIG)).run(("emit_l2b",), clean=False)
    rows = []
    for v in VARIANTS:
        for name in AOIS:
            run(name, v)
        rows.append({"variant": v, **score_thanewasna(), **score_kehalpur()})
        print(f"[done] {v}")
    df = pd.DataFrame(rows).set_index("variant").T
    df.to_csv(out / "results.csv")
    pd.set_option("display.width", 200)
    print(df.to_string())
    for name in AOIS:  # leave the baseline in place
        run(name, "none")


if __name__ == "__main__":
    main()
