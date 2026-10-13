"""Geological-consistency comparison of legacy vs geoscan CPI when no mineralisation locations exist.

    python scripts/compare_lithology.py --name kehalpur

This is NOT a validation against known copper. It asks whether each method ranks the
lithologies that can host the mineralisation above those that cannot, using the GSI map and the
host-rock statements of the GSI report (CRO-19966-1987, Kherli area):

    host      metapelite, calc-silicate/metapelite (main ore host), granitic rocks and granite
              veins ("meta igneous rocks of granitic composition ... can be considered to host")
    non-host  amphibolite meta-basalt, pillow basalt, meta-gabbro, meta-ultramafic
              (non-economic or barren)
    cover     Deccan trap, infra-trap sediments (Cretaceous; younger than the mineralisation)

Contrast A: Precambrian basement (host + non-host) vs cover.   Contrast B: host vs non-host.

Inputs: data/<name>/aoi/*.geojson, data/<name>/ground_truth/lithology_gsi_cro19966.geojson,
        outputs/<name>_legacy/ (legacy outputs3 copy), outputs/<name>/ (geoscan).
Writes outputs/<name>_comparison/.
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
from pyproj import Transformer  # noqa: E402
from rasterio.features import rasterize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_methods import geoscan_layers, legacy_layers  # noqa: E402
from geoscan.aoi import load_aoi  # noqa: E402
from geoscan.config import TargetConfig  # noqa: E402
from geoscan.grid import grid_for_aoi  # noqa: E402
from geoscan.prospectivity.targets import select_targets  # noqa: E402
from geoscan.raster_io import aoi_mask, write_raster  # noqa: E402
from geoscan.validation import bootstrap_auc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COVER, NONHOST, HOST = 1, 2, 3
GROUP_NAMES = {COVER: "cover (Deccan / infra-trap)", NONHOST: "basement non-host (mafic/ultramafic)",
               HOST: "basement host (metapelite / granitic)"}


def lith_group(unit: str) -> int:
    u = " ".join(str(unit).lower().split())
    if "deccan" in u or "infra trap" in u:
        return COVER
    if "metapelite" in u or "granit" in u:
        return HOST
    return NONHOST  # amphibolite meta basalt / meta gabbro, meta ultramafic, pillow structure


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="kehalpur")
    args = ap.parse_args(argv)
    name = args.name
    data, out = ROOT / "data" / name, ROOT / "outputs" / f"{name}_comparison"
    for d in ("tables", "maps", "rasters"):
        (out / d).mkdir(parents=True, exist_ok=True)

    aoi = load_aoi(next((data / "aoi").glob("*.geojson")), name)
    grid = grid_for_aoi(aoi, 15.0, 180.0)
    aoi_utm = aoi.to_crs(grid.crs)
    inside = aoi_mask(grid, aoi_utm)

    lith = gpd.read_file(data / "ground_truth/lithology_gsi_cro19966.geojson").to_crs(grid.crs)
    lith["group"] = lith["LITHOLOGICAL_UNIT"].map(lith_group)
    cls = np.zeros(grid.shape, dtype="uint8")
    for g in (COVER, NONHOST, HOST):  # later groups overwrite earlier ones (small host bodies win)
        sub = lith[lith["group"] == g]
        if len(sub):
            burn = rasterize([(geom, 1) for geom in sub.geometry], out_shape=grid.shape, transform=grid.transform,
                             dtype="uint8").astype(bool)
            cls[burn] = g
    cls[~inside] = 0
    areas = {GROUP_NAMES[g]: round(float((cls == g).sum()) * grid.res ** 2 / 1e6, 2) for g in GROUP_NAMES}
    unmapped = round(float(((cls == 0) & inside).sum()) * grid.res ** 2 / 1e6, 2)

    L = legacy_layers(ROOT / "outputs" / f"{name}_legacy", grid, inside)
    G = geoscan_layers(ROOT / "outputs" / name, grid, inside)
    surfaces = {"legacy_as_delivered": L["cpi_file"], "legacy_flip_fixed": L["cpi_fixed"], "geoscan": G["cpi"]}
    components = {**{f"legacy_{k}": L[k] for k in ("muscovite", "chlorite", "gossan", "quartz", "structure")},
                  **{f"geoscan_{k}": G[k] for k in ("muscovite", "chlorite", "gossan", "quartz", "structure")}}
    write_raster(out / "rasters/cpi_surfaces_15m.tif", np.stack(list(surfaces.values())), grid, list(surfaces))
    write_raster(out / "rasters/lithology_groups_15m.tif", cls.astype("float32"), grid, ["group"],
                 nodata=0, tags={k: v for k, v in zip(["1", "2", "3"], GROUP_NAMES.values())})

    rng = np.random.default_rng(0)

    supported = np.nan_to_num(G["data_support"]) >= 0.20  # where geoscan had usable EMIT pixels

    def contrast(arr, pos_mask, neg_mask, region=None):
        v = inside & np.isfinite(arr) & (region if region is not None else True)
        pos, neg = arr[v & pos_mask], arr[v & neg_mask]
        pos = rng.choice(pos, min(pos.size, 3000), replace=False) if pos.size else pos
        neg = rng.choice(neg, min(neg.size, 3000), replace=False) if neg.size else neg
        return bootstrap_auc(pos, neg, n=1000)

    basement = (cls == HOST) | (cls == NONHOST)
    rows = []
    for key, arr in {**surfaces, **components}.items():
        a = contrast(arr, basement, cls == COVER)
        b = contrast(arr, cls == HOST, cls == NONHOST)
        means = {f"mean_{GROUP_NAMES[g].split(' (')[0].replace(' ', '_')}": round(float(np.nanmean(arr[(cls == g) & inside])), 4)
                 for g in GROUP_NAMES}
        a2 = contrast(arr, basement, cls == COVER, supported)
        b2 = contrast(arr, cls == HOST, cls == NONHOST, supported)
        rows.append({"layer": key, "A_basement_vs_cover_auc": a["auc"], "A_ci": f"{a['ci_low']}-{a['ci_high']}",
                     "B_host_vs_nonhost_auc": b["auc"], "B_ci": f"{b['ci_low']}-{b['ci_high']}",
                     "A_auc_supported_area": a2["auc"], "B_auc_supported_area": b2["auc"], **means})
    res = pd.DataFrame(rows)
    res.to_csv(out / "tables/lithology_contrast_auc.csv", index=False)

    # targets: which lithology do they fall on?
    to_utm = Transformer.from_crs(4326, grid.crs, always_xy=True)
    lt = pd.read_csv(ROOT / "outputs" / f"{name}_legacy" / "csv/top50_unmixed_targets_summary.csv")
    lx, ly = to_utm.transform(lt["Longitude"].values, lt["Latitude"].values)
    fixed = select_targets(L["cpi_fixed"], grid, {}, inside & np.isfinite(L["cpi_fixed"]), TargetConfig())
    gt = pd.read_csv(ROOT / "outputs" / name / "tables/top50_targets.csv")
    tsets = {"legacy_as_delivered": np.column_stack([lx, ly]),
             "legacy_flip_fixed": fixed[["Easting", "Northing"]].to_numpy(),
             "geoscan": gt[["Easting", "Northing"]].to_numpy()}
    share = {GROUP_NAMES[g]: (cls == g).sum() / inside.sum() for g in GROUP_NAMES}
    trows = []
    for key, xy in tsets.items():
        c = np.clip(((xy[:, 0] - grid.x0) / grid.res).astype(int), 0, grid.width - 1)
        r = np.clip(((grid.y0 - xy[:, 1]) / grid.res).astype(int), 0, grid.height - 1)
        g = cls[r, c]
        row = {"method": key, "n_targets": len(xy)}
        for code, label in GROUP_NAMES.items():
            short = label.split(" (")[0]
            row[f"{short}_targets"] = int((g == code).sum())
            row[f"{short}_expected_by_area"] = round(float(share[label] * len(xy)), 1)
        trows.append(row)
    tdf = pd.DataFrame(trows)
    tdf.to_csv(out / "tables/targets_by_lithology.csv", index=False)

    # figure: three CPI panels with lithology outlines and top-10 targets
    ll = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    xs = grid.x0 + (np.arange(grid.width) + 0.5) * grid.res
    ys = grid.y0 - (np.arange(grid.height) + 0.5) * grid.res
    LON, LAT = ll.transform(*np.meshgrid(xs, ys))
    lith_ll = lith.to_crs(4326)
    colors = {COVER: "#00e5ff", NONHOST: "#bdbdbd", HOST: "#76ff03"}
    titles = {"legacy_as_delivered": "Legacy CPI — as delivered", "legacy_flip_fixed": "Legacy CPI — flip corrected",
              "geoscan": "geoscan CPI (Phase 1)"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 20), dpi=150)
    r = res.set_index("layer")
    for ax, (key, arr) in zip(axes, surfaces.items()):
        vals = arr[inside & np.isfinite(arr)]
        lo, hi = np.percentile(vals, [2, 98])
        cf = ax.contourf(LON, LAT, np.clip(arr, lo, hi), levels=30, cmap="plasma")
        for g, col in colors.items():
            sub = lith_ll[lith_ll["group"] == g]
            if len(sub):
                sub.boundary.plot(ax=ax, color=col, linewidth=1.1 if g != NONHOST else 0.6, zorder=5)
        xy = tsets[key][:10]
        tlon, tlat = ll.transform(xy[:, 0], xy[:, 1])
        ax.scatter(tlon, tlat, s=110, marker="*", c="white", edgecolors="#d50000", linewidths=1.3, zorder=8)
        x, y = aoi.geometry.exterior.xy
        ax.plot(x, y, color="black", lw=1.5)
        ax.set_xlim(aoi.bounds[0], aoi.bounds[2])
        ax.set_ylim(aoi.bounds[1], aoi.bounds[3])
        ax.set_aspect(1 / np.cos(np.radians(np.mean(y))))
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_title(f"{titles[key]} — basement vs cover AUC {r.loc[key, 'A_basement_vs_cover_auc']:.2f} · "
                     f"host vs non-host AUC {r.loc[key, 'B_host_vs_nonhost_auc']:.2f}", fontsize=10, fontweight="bold")
        fig.colorbar(cf, ax=ax, shrink=0.8, label="CPI (2-98% stretch)")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([0], [0], color=c, lw=2, label=GROUP_NAMES[g]) for g, c in colors.items()] +
                   [Line2D([0], [0], marker="*", color="w", markerfacecolor="white", markeredgecolor="#d50000",
                           markersize=12, label="Top-10 targets")], loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{name.title()} — CPI methods vs GSI lithology (consistency check, not copper validation)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "maps/cpi_methods_vs_lithology.png")
    plt.close(fig)

    lines = [f"# {name.title()} — legacy vs geoscan CPI: geological-consistency check\n",
             "**Not a validation against copper.** No borehole or lode locations exist in the NGDR data for this "
             "block, so each CPI is compared with the GSI lithology map and the report's host-rock statements.\n",
             "Areas in block (km²): " + ", ".join(f"{k}: {v}" for k, v in areas.items()) + f", unmapped: {unmapped}\n",
             f"Area with usable EMIT data (geoscan support >= 0.20): "
             f"{round(float((supported & inside).sum()) * grid.res ** 2 / 1e6, 1)} km²\n",
             "| Method | A: basement vs cover AUC (95% CI) | B: host vs non-host AUC (95% CI) | A / B on EMIT-supported "
             "area only | Top-50 targets on cover (expected by area) | on host (expected) |",
             "|---|---|---|---|---|---|"]
    t = tdf.set_index("method")
    for key in surfaces:
        lines.append(f"| {titles[key]} | {r.loc[key, 'A_basement_vs_cover_auc']:.2f} ({r.loc[key, 'A_ci']}) "
                     f"| {r.loc[key, 'B_host_vs_nonhost_auc']:.2f} ({r.loc[key, 'B_ci']}) "
                     f"| {r.loc[key, 'A_auc_supported_area']:.2f} / {r.loc[key, 'B_auc_supported_area']:.2f} "
                     f"| {t.loc[key, 'cover_targets']}/{t.loc[key, 'n_targets']} ({t.loc[key, 'cover_expected_by_area']}) "
                     f"| {t.loc[key, 'basement host_targets']}/{t.loc[key, 'n_targets']} "
                     f"({t.loc[key, 'basement host_expected_by_area']}) |")
    lines += ["\n| Component | A AUC | B AUC |", "|---|---|---|"]
    for key in components:
        lines.append(f"| {key} | {r.loc[key, 'A_basement_vs_cover_auc']:.2f} | {r.loc[key, 'B_host_vs_nonhost_auc']:.2f} |")
    lines += ["\nAUC > 0.5 = higher CPI on basement (A) or host rocks (B). Granitic rocks cover about half the block, so "
              "contrast B is weak by construction; the GSI map is 1:~25k and its contacts are approximate."]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
