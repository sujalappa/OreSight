"""Compare the legacy pipeline (outputs3-style) with geoscan against known copper mineralisation.

    python scripts/compare_methods.py --name thanewasna

Inputs (under phase-1-copper-exploration/):
    data/<name>/aoi/*.geojson                         block polygon
    data/<name>/ground_truth/cu_lodes_mineral_zone.geojson, Thanewasna_Borehole_Ground_Truth_Master.csv
    outputs/<name>_legacy/                            legacy scripts' outputs (run unchanged)
    outputs/<name>/                                   geoscan outputs
Writes outputs/<name>_comparison/{tables,maps}/ and summary.md.

Three CPI surfaces are scored identically on one 15 m UTM grid:
    legacy_as_delivered  legacy CPI as shown on its maps / used for its targets (includes the
                         N-S layer misregistration of the legacy code)
    legacy_flip_fixed    legacy formula recomputed from correctly registered legacy layers
    geoscan              Phase-1 geoscan CPI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoscan.aoi import load_aoi  # noqa: E402
from geoscan.config import TargetConfig  # noqa: E402
from geoscan.grid import grid_for_aoi  # noqa: E402
from geoscan.prospectivity.targets import select_targets  # noqa: E402
from geoscan.raster_io import aoi_mask, write_raster  # noqa: E402
from geoscan.validation import bootstrap_auc, captured_at, success_rate, target_hits  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def to_grid(arr, transform, crs, grid, resampling=Resampling.bilinear):
    dst = np.full(grid.shape, np.nan, "float32")
    reproject(np.asarray(arr, "float32"), dst, src_transform=transform, src_crs=crs, dst_transform=grid.transform,
              dst_crs=grid.crs, resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def read(path, band=1):
    with rasterio.open(path) as s:
        a = s.read(band).astype("float32")
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs


def legacy_layers(ldir: Path, grid, inside_eval):
    """Legacy layers on the eval grid. EMIT-derived legacy GeoTIFFs hold south-up arrays in a
    north-up file, so np.flipud restores their true position (= what the legacy maps show)."""
    r = ldir / "rasters"
    out, raw = {}, {}
    for key, fname in (("muscovite", "mineral_abundance_muscovite.tif"), ("chlorite", "mineral_abundance_chlorite.tif"),
                       ("gossan", "mineral_abundance_gossan.tif"), ("cpi_file", "unmixed_copper_prospectivity_cpi.tif")):
        a, t, c = read(r / fname)
        raw[key] = (np.flipud(a), t, c)
    q, tq, cq = read(r / "mineral_abundance_quartz.tif")          # quartz array is north-up already
    s, ts, cs = read(r / "structural_lineaments_straight_proximity.tif")  # north-up, correct
    raw["quartz"], raw["structure"] = (q, tq, cq), (s, ts, cs)

    # legacy formula on the legacy 15 m grid, with all layers correctly registered
    a_m, a_c, a_g = raw["muscovite"][0], raw["chlorite"][0], raw["gossan"][0]
    h = 0.30 * a_m + 0.28 * q + 0.22 * a_g + 0.20 * a_c
    cpi = gaussian_filter(h * (1 + 0.40 * np.nan_to_num(s)), 1.0)
    cpi = (cpi - cpi.min()) / (cpi.max() - cpi.min() + 1e-12)
    raw["cpi_fixed"] = (cpi, raw["cpi_file"][1], raw["cpi_file"][2])

    for k, (a, t, c) in raw.items():
        out[k] = np.where(inside_eval, to_grid(a, t, c, grid), np.nan)
    return out


def geoscan_layers(gdir: Path, grid, inside_eval):
    out = {}
    with rasterio.open(gdir / "rasters/targets/cpi.tif") as s:
        names = list(s.descriptions)
        for band, key in (("cpi", "cpi"), ("structural_proximity", "structure")):
            out[key] = s.read(names.index(band) + 1).astype("float32"), s.transform, s.crs
    with rasterio.open(gdir / "rasters/alteration/alteration_stack_30m.tif") as s:
        names = list(s.descriptions)
        for key in ("muscovite", "chlorite", "gossan", "quartz", "data_support"):
            out[key] = s.read(names.index(key) + 1).astype("float32"), s.transform, s.crs
    return {k: np.where(inside_eval, to_grid(a, t, c, grid), np.nan) for k, (a, t, c) in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="thanewasna")
    args = ap.parse_args(argv)
    name = args.name
    data, out = ROOT / "data" / name, ROOT / "outputs" / f"{name}_comparison"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(parents=True, exist_ok=True)

    aoi = load_aoi(next((data / "aoi").glob("*.geojson")), name)
    grid = grid_for_aoi(aoi, 15.0, 180.0)
    aoi_utm = aoi.to_crs(grid.crs)
    inside = aoi_mask(grid, aoi_utm)
    to_utm = Transformer.from_crs(4326, grid.crs, always_xy=True)

    lodes = gpd.read_file(data / "ground_truth/cu_lodes_mineral_zone.geojson").to_crs(grid.crs)
    lode_cells = rasterize([(g, 1) for g in lodes.geometry], out_shape=grid.shape, transform=grid.transform,
                           all_touched=True, dtype="uint8").astype(bool) & inside
    bh = pd.read_csv(data / "ground_truth/Thanewasna_Borehole_Ground_Truth_Master.csv")
    bh["lode"] = bh["Intersected_Cu_Lode"].str.lower().str.startswith("lode")
    bx, by = to_utm.transform(bh["BH_Lon"].values, bh["BH_Lat"].values)
    bcol = np.clip(((bx - grid.x0) / grid.res).astype(int), 0, grid.width - 1)
    brow = np.clip(((grid.y0 - by) / grid.res).astype(int), 0, grid.height - 1)

    L = legacy_layers(ROOT / "outputs" / f"{name}_legacy", grid, inside)
    G = geoscan_layers(ROOT / "outputs" / name, grid, inside)
    surfaces = {"legacy_as_delivered": L["cpi_file"], "legacy_flip_fixed": L["cpi_fixed"], "geoscan": G["cpi"]}
    components = {**{f"legacy_{k}": L[k] for k in ("muscovite", "chlorite", "gossan", "quartz", "structure")},
                  **{f"geoscan_{k}": G[k] for k in ("muscovite", "chlorite", "gossan", "quartz", "structure")}}
    write_raster(out / "rasters/cpi_surfaces_15m.tif", np.stack(list(surfaces.values())), grid, list(surfaces))

    # ---------- A. boreholes (collar value) ----------
    rows = []
    for key, arr in {**surfaces, **components}.items():
        v = arr[brow, bcol]
        rows.append({"layer": key, **bootstrap_auc(v[bh["lode"].values], v[~bh["lode"].values])})
    bh_auc = pd.DataFrame(rows)
    bh_auc.to_csv(out / "tables/borehole_auc.csv", index=False)

    # ---------- B. lode polygons (pixel level) ----------
    rows, curves = [], []
    rng = np.random.default_rng(1)
    for key, arr in {**surfaces, **components}.items():
        valid = inside & np.isfinite(arr)
        pos, neg = arr[valid & lode_cells], arr[valid & ~lode_cells]
        # subsample negatives for the permutation test (spatially correlated anyway)
        neg_s = rng.choice(neg, min(neg.size, 3000), replace=False) if neg.size else neg
        res = bootstrap_auc(pos, neg_s, n=1000)
        curve = success_rate(arr, lode_cells, valid)
        rows.append({"layer": key, **res, **captured_at(curve)})
        if key in surfaces:
            curves.append(curve.assign(method=key))
    lode_auc = pd.DataFrame(rows)
    lode_auc.to_csv(out / "tables/lode_pixel_auc.csv", index=False)
    pd.concat(curves).iloc[::5].to_csv(out / "tables/success_rate_curves.csv", index=False)

    # ---------- C. targets ----------
    lt = pd.read_csv(ROOT / "outputs" / f"{name}_legacy" / "csv/top50_unmixed_targets_summary.csv")
    lx, ly = to_utm.transform(lt["Longitude"].values, lt["Latitude"].values)
    fixed = select_targets(L["cpi_fixed"], grid, {}, inside & np.isfinite(L["cpi_fixed"]), TargetConfig())
    gt = pd.read_csv(ROOT / "outputs" / name / "tables/top50_targets.csv")
    tsets = {"legacy_as_delivered": np.column_stack([lx, ly]),
             "legacy_flip_fixed": fixed[["Easting", "Northing"]].to_numpy() if len(fixed) else np.zeros((0, 2)),
             "geoscan": gt[["Easting", "Northing"]].to_numpy()}
    # top-10 hits as well as all targets
    trows = []
    for key, xy in tsets.items():
        for label, sub in (("all", xy), ("top10", xy[:10])):
            trows.append({"method": key, "subset": label, **target_hits(sub, lodes.geometry, aoi_utm)})
    hits = pd.DataFrame(trows)
    hits.to_csv(out / "tables/target_hits.csv", index=False)

    # ---------- figures ----------
    ll = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    xs = grid.x0 + (np.arange(grid.width) + 0.5) * grid.res
    ys = grid.y0 - (np.arange(grid.height) + 0.5) * grid.res
    X, Y = np.meshgrid(xs, ys)
    LON, LAT = ll.transform(X, Y)
    lodes_ll = lodes.to_crs(4326)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.2), dpi=180, sharey=True)
    titles = {"legacy_as_delivered": "Legacy CPI — as delivered", "legacy_flip_fixed": "Legacy CPI — flip corrected",
              "geoscan": "geoscan CPI (Phase 1)"}
    for ax, (key, arr) in zip(axes, surfaces.items()):
        vals = arr[inside & np.isfinite(arr)]
        lo, hi = np.percentile(vals, [2, 98])
        cf = ax.contourf(LON, LAT, np.clip(arr, lo, hi), levels=30, cmap="plasma")
        ax.contour(LON, LAT, arr, levels=np.percentile(vals, [80, 90]), colors=["white", "#ffeb3b"], linewidths=0.9)
        lodes_ll.plot(ax=ax, facecolor="none", edgecolor="#00e676", linewidth=1.3, zorder=5)
        ax.scatter(bh.loc[~bh.lode, "BH_Lon"], bh.loc[~bh.lode, "BH_Lat"], s=12, c="#bdbdbd", edgecolors="black",
                   linewidths=0.4, zorder=6, label="Borehole: halo only")
        ax.scatter(bh.loc[bh.lode, "BH_Lon"], bh.loc[bh.lode, "BH_Lat"], s=30, c="#ff1744", edgecolors="black",
                   linewidths=0.6, marker="D", zorder=7, label="Borehole: Cu lode")
        txy = tsets[key][:10]
        if len(txy):
            tlon, tlat = ll.transform(txy[:, 0], txy[:, 1])
            ax.scatter(tlon, tlat, s=90, marker="*", c="white", edgecolors="#d50000", linewidths=1.2, zorder=8,
                       label="Top-10 targets")
        la = lode_auc.set_index("layer").loc[key]
        ba = bh_auc.set_index("layer").loc[key]
        ax.set_title(f"{titles[key]}\nlode-pixel AUC {la['auc']:.2f} · borehole AUC {ba['auc']:.2f}\n"
                     f"top 20% of area captures {100 * la['captured_top20pct']:.0f}% of lodes", fontsize=9.5,
                     fontweight="bold")
        x, y = aoi.geometry.exterior.xy
        ax.plot(x, y, color="black", lw=1.4)
        ax.set_aspect(1 / np.cos(np.radians(np.mean(y))))
        ax.set_xlabel("Longitude (°E)")
        ax.ticklabel_format(useOffset=False, style="plain")
        fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08, shrink=0.8, label="CPI (2-98% stretch)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle(f"{name.title()} — CPI methods vs known copper lodes (green outlines) and boreholes",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "maps/cpi_methods_vs_lodes.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    for c in curves:
        ax.plot(100 * c["area_fraction"], 100 * c["captured_fraction"], lw=2, label=titles[c["method"].iloc[0]])
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="Random")
    ax.set_xlabel("Block area ranked by CPI, high to low (%)")
    ax.set_ylabel("Known lode area captured (%)")
    ax.set_title("Success-rate curves (higher and further left = better)", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "maps/success_rate_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    comp = lode_auc.copy()
    comp["x"] = np.arange(len(comp))
    colors = ["#7b1fa2" if k.startswith("legacy") else "#00897b" for k in comp["layer"]]
    ax.bar(comp["x"], comp["auc"] - 0.5, bottom=0.5, color=colors,
           yerr=[comp["auc"] - comp["ci_low"], comp["ci_high"] - comp["auc"]], capsize=3)
    ax.axhline(0.5, color="black", lw=1)
    ax.set_xticks(comp["x"], comp["layer"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("AUC inside lodes vs rest of block")
    ax.set_ylim(0.2, 1.0)
    ax.set_title("Which layer separates lode pixels from the rest? (0.5 = no skill; bars = 95% CI)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "maps/layer_auc_lodes.png")
    plt.close(fig)

    # ---------- summary ----------
    s = lode_auc.set_index("layer")
    b = bh_auc.set_index("layer")
    h = hits.set_index(["method", "subset"])
    lines = [f"# {name.title()} — legacy vs geoscan CPI against known copper\n",
             f"Block {aoi.area_km2():.2f} km² · {len(lodes)} lode polygons ({lodes.area.sum() / 1e6:.3f} km², "
             f"{100 * lode_cells.sum() / inside.sum():.1f}% of block) · {len(bh)} boreholes "
             f"({int(bh.lode.sum())} intersect a Cu lode)\n",
             "| Method | Lode-pixel AUC (95% CI) | p | Top-20% area captures | Borehole AUC (95% CI) | p | "
             "Top-10 targets ≤100 m of lode (random) | All targets ≤100 m (random) |",
             "|---|---|---|---|---|---|---|---|"]
    for key in surfaces:
        lines.append(f"| {titles[key]} | {s.loc[key, 'auc']:.2f} ({s.loc[key, 'ci_low']:.2f}–{s.loc[key, 'ci_high']:.2f}) "
                     f"| {s.loc[key, 'p_value']:.3f} | {100 * s.loc[key, 'captured_top20pct']:.0f}% "
                     f"| {b.loc[key, 'auc']:.2f} ({b.loc[key, 'ci_low']:.2f}–{b.loc[key, 'ci_high']:.2f}) "
                     f"| {b.loc[key, 'p_value']:.3f} "
                     f"| {h.loc[(key, 'top10'), 'hits_within_100m']} ({h.loc[(key, 'top10'), 'expected_random_100m']}) "
                     f"| {h.loc[(key, 'all'), 'hits_within_100m']}/{h.loc[(key, 'all'), 'n_targets']} "
                     f"({h.loc[(key, 'all'), 'expected_random_100m']}) |")
    lines += ["\nComponent layers (lode-pixel AUC):\n", "| Layer | AUC | 95% CI | Borehole AUC |", "|---|---|---|---|"]
    for key in components:
        lines.append(f"| {key} | {s.loc[key, 'auc']:.2f} | {s.loc[key, 'ci_low']:.2f}–{s.loc[key, 'ci_high']:.2f} "
                     f"| {b.loc[key, 'auc']:.2f} |")
    lines += ["\nNotes: AUC 0.5 = no better than random. Lode polygons are plan outlines of sub-surface lodes; "
              "boreholes are inclined (~60°), so collars are offset from where they cut the lode. "
              "Spatial autocorrelation makes pixel-level p-values optimistic."]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
