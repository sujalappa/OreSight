"""Validation of the geoscan Phase-1 results against independent ground truth.

    python scripts/validate.py --name thanewasna     # boreholes + Cu lode polygons
    python scripts/validate.py --name kehalpur        # GSI lithology + NGDR exploration-block footprints

Writes outputs/<name>_validation/{summary.md, tables/, maps/}.

Thanewasna  CPI and each layer vs 30 Cu lode polygons (pixel AUC, success-rate curve), CPI at 47
            borehole collars (8 lode-intersecting vs 39 halo), targets within 100 m of a lode.
Kehalpur     no borehole / lode coordinates are available, so: CPI on Precambrian basement vs
            Deccan / infra-trap cover (GSI CRO-19966-1987 lithology) and CPI inside NGDR
            exploration-block footprints (CRO-20859-1986, CRO-23431-2018, CRO-22666-2009,
            CRO-21815-1993) vs the rest of the block. These are consistency checks, not copper validation.
"""

from __future__ import annotations

import argparse
import json
import os
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
from shapely.geometry import Point  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from compare_lithology import COVER, HOST, NONHOST, lith_group  # noqa: E402
from compare_methods import geoscan_layers  # noqa: E402
from geoscan.aoi import load_aoi  # noqa: E402
from geoscan.grid import grid_for_aoi  # noqa: E402
from geoscan.raster_io import aoi_mask  # noqa: E402
from geoscan.validation import bootstrap_auc, captured_at, success_rate, target_hits, zone_capture  # noqa: E402

ROOT = Path(os.environ.get("GEOSCAN_ROOT") or HERE.parents[1])
LAYERS = ("cpi", "gossan", "muscovite", "chlorite", "quartz", "structure")
LABEL = {"cpi": "CPI", "gossan": "Gossan / iron oxide", "muscovite": "Muscovite / sericite",
         "chlorite": "Chlorite / epidote", "quartz": "Quartz / silicification", "structure": "Structural proximity"}
NGDR = {"CRO-20859-1986": "Bhawratekra detailed exploration (G3)",
        "CRO-23431-2018": "Banskhapa–Pipariya general exploration (G2)",
        "CRO-22666-2009": "Banskhapa–Pipariya mineralised / alteration zones",
        "CRO-21815-1993": "Bargaon–Tarora sulphide investigation"}
rng = np.random.default_rng(0)
OUTPUT = None   # outputs/<OUTPUT> to validate (default: outputs/<block>)


def _auc(arr, pos, neg, valid, cap=3000, n=1000):
    v = valid & np.isfinite(arr)
    p, q = arr[v & pos], arr[v & neg]
    p = rng.choice(p, min(cap, p.size), replace=False) if p.size > cap else p
    q = rng.choice(q, min(cap, q.size), replace=False) if q.size > cap else q
    return bootstrap_auc(p, q, n=n)


def _zone_masks(zones, G, grid, inside, tier3):
    """Boolean rasters for the nested zone sets (None when the run has no zone file)."""
    if zones is None:
        return None

    def burn(sub):
        return rasterize([(g, 1) for g in sub.geometry], out_shape=grid.shape, transform=grid.transform,
                         all_touched=True, dtype="uint8").astype(bool) & inside if len(sub) else np.zeros(grid.shape, bool)

    t = zones["Priority_Tier"].str.extract(r"Tier (\d)")[0].astype(int)
    return {"Tier 1 zones": burn(zones[t == 1]), "Tier 1 + 2 zones": burn(zones[t <= 2]),
            "All zones (Tier 1-3)": burn(zones),
            f"Outer halo (all ground with CPI >= {100 * tier3:.0f} %)": inside & (np.nan_to_num(G["cpi"]) >= tier3)}


def _setup(name):
    aoi = load_aoi(next(p for p in (ROOT / "data" / name / "aoi").glob("*.geojson") if p.name != "aoi.geojson"), name)
    grid = grid_for_aoi(aoi, 15.0, 180.0)
    aoi_utm = aoi.to_crs(grid.crs)
    inside = aoi_mask(grid, aoi_utm)
    src = ROOT / "outputs" / (OUTPUT or name)
    G = geoscan_layers(src, grid, inside)
    loc = src / "tables/field_check_locations.csv"
    targets = pd.read_csv(loc if loc.exists() else src / "tables/top50_targets.csv")
    zp = src / "vectors/prospective_zones.geojson"
    zones = gpd.read_file(zp).to_crs(grid.crs) if zp.exists() else None
    tier3 = json.loads((src / "manifest.json").read_text(encoding="utf-8"))["settings"]["targets"]["tier3"]
    return aoi, grid, aoi_utm, inside, G, targets, zones, _zone_masks(zones, G, grid, inside, tier3)


def _panel(ax, grid, arr, inside, aoi, title):
    ll = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    xs = grid.x0 + (np.arange(grid.width) + 0.5) * grid.res
    ys = grid.y0 - (np.arange(grid.height) + 0.5) * grid.res
    LON, LAT = ll.transform(*np.meshgrid(xs, ys))
    vals = 100 * arr[inside & np.isfinite(arr)]
    cf = ax.contourf(LON, LAT, np.clip(100 * arr, 0, 100), levels=np.linspace(0, 100, 41), cmap="plasma")
    ax.contour(LON, LAT, 100 * arr, levels=[28, 50, 70], colors=["white", "#ffeb3b", "#ff1744"], linewidths=1.0)
    x, y = aoi.geometry.exterior.xy
    ax.plot(x, y, color="black", lw=1.5)
    ax.set_xlim(aoi.bounds[0], aoi.bounds[2])
    ax.set_ylim(aoi.bounds[1], aoi.bounds[3])
    ax.set_aspect(1 / np.cos(np.radians(np.mean(y))))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    return cf


def thanewasna(out: Path):
    name = "thanewasna"
    aoi, grid, aoi_utm, inside, G, targets, zones, zmask = _setup(name)
    gt = ROOT / "data" / name / "ground_truth"
    lodes = gpd.read_file(gt / "cu_lodes_mineral_zone.geojson").to_crs(grid.crs)
    lode = rasterize([(g, 1) for g in lodes.geometry], out_shape=grid.shape, transform=grid.transform,
                     all_touched=True, dtype="uint8").astype(bool) & inside
    bh = pd.read_csv(gt / "Thanewasna_Borehole_Ground_Truth_Master.csv")
    bh["lode"] = bh["Intersected_Cu_Lode"].str.lower().str.startswith("lode")
    bx, by = Transformer.from_crs(4326, grid.crs, always_xy=True).transform(bh.BH_Lon.values, bh.BH_Lat.values)
    bc = np.clip(((bx - grid.x0) / grid.res).astype(int), 0, grid.width - 1)
    br = np.clip(((grid.y0 - by) / grid.res).astype(int), 0, grid.height - 1)

    rows = []
    for k in LAYERS:
        a = G[k]
        lo = _auc(a, lode, ~lode, inside)
        b = bootstrap_auc(a[br, bc][bh.lode.values], a[br, bc][~bh.lode.values], n=2000)
        cap = captured_at(success_rate(a, lode, inside & np.isfinite(a)))
        rows.append({"layer": LABEL[k], "lode_auc": lo["auc"], "lode_ci": f"{lo['ci_low']:.2f}–{lo['ci_high']:.2f}",
                     "lode_p": lo["p_value"], "borehole_auc": b["auc"],
                     "borehole_ci": f"{b['ci_low']:.2f}–{b['ci_high']:.2f}", "borehole_p": b["p_value"], **cap})
    res = pd.DataFrame(rows)
    res.to_csv(out / "tables/validation_auc.csv", index=False)
    xy = targets[["Easting", "Northing"]].to_numpy()
    hits = pd.DataFrame([{"subset": "top 10", **target_hits(xy[:10], lodes.geometry, aoi_utm)},
                         {"subset": "all", **target_hits(xy, lodes.geometry, aoi_utm)}])
    hits.to_csv(out / "tables/field_check_location_hits.csv", index=False)
    zcap = None
    if zmask:
        zcap = zone_capture(zmask, lode, inside)
        zcap["lode_boreholes_inside"] = [f"{int(m[br, bc][bh.lode.values].sum())} of {int(bh.lode.sum())}" for m in zmask.values()]
        zcap["halo_boreholes_inside"] = [f"{int(m[br, bc][~bh.lode.values].sum())} of {int((~bh.lode).sum())}" for m in zmask.values()]
        zcap.to_csv(out / "tables/zone_capture.csv", index=False)
    curve = success_rate(G["cpi"], lode, inside & np.isfinite(G["cpi"]))
    curve.iloc[::5].to_csv(out / "tables/success_rate_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 9), dpi=180)
    cf = _panel(ax, grid, G["cpi"], inside, aoi, "Thanewasna — CPI vs known Cu lodes and boreholes")
    lodes.to_crs(4326).plot(ax=ax, facecolor="none", edgecolor="#00e676", linewidth=1.4, zorder=5)
    ax.scatter(bh.loc[~bh.lode, "BH_Lon"], bh.loc[~bh.lode, "BH_Lat"], s=14, c="#bdbdbd", edgecolors="black",
               linewidths=0.4, zorder=6, label="Borehole: halo only")
    ax.scatter(bh.loc[bh.lode, "BH_Lon"], bh.loc[bh.lode, "BH_Lat"], s=34, c="#ff1744", edgecolors="black",
               marker="D", zorder=7, label="Borehole: intersects Cu lode")
    if zones is not None and len(zones):
        zones.to_crs(4326).boundary.plot(ax=ax, color="#00e5ff", linewidth=1.8, zorder=7)
        ax.plot([], [], color="#00e5ff", lw=2, label="Prospective zone")
    ax.scatter(targets.Longitude[:10], targets.Latitude[:10], s=45, marker="*", c="#ffd600", edgecolors="black",
               linewidths=0.6, zorder=8, label="Suggested field-check locations (top 10)")
    ax.plot([], [], color="#00e676", lw=2, label="Cu lode (MECL Mineral_Zone_DM)")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08, shrink=0.8, label="CPI (%)")
    fig.tight_layout()
    fig.savefig(out / "maps/cpi_vs_lodes_boreholes.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    ax.plot(100 * curve.area_fraction, 100 * curve.captured_fraction, lw=2.2, color="#6a1b9a", label="CPI")
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="Random")
    ax.set_xlabel("Block area ranked by CPI, high to low (%)")
    ax.set_ylabel("Known Cu lode area captured (%)")
    ax.set_title("Thanewasna — success-rate curve", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "maps/success_rate_curve.png")
    plt.close(fig)

    c = res.iloc[0]
    h = hits.set_index("subset")
    lines = ["# Thanewasna — validation against known copper mineralisation\n",
             f"Ground truth: {len(lodes)} Cu lode polygons (MECL `Mineral_Zone_DM`, {lodes.area.sum() / 1e6:.3f} km², "
             f"{100 * lode.sum() / inside.sum():.1f}% of the block) and {len(bh)} borehole collars "
             f"({int(bh.lode.sum())} intersect a lode at 0.20% or 0.50% Cu cut-off; {int((~bh.lode).sum())} halo only).\n",
             f"- **CPI vs lode area:** AUC {c.lode_auc:.2f} (95% CI {c.lode_ci}); the top 10 / 20 / 30% of the block "
             f"by CPI contains {100 * c.captured_top10pct:.0f} / {100 * c.captured_top20pct:.0f} / "
             f"{100 * c.captured_top30pct:.0f}% of the lode area.",
             f"- **CPI at borehole collars:** AUC {c.borehole_auc:.2f} (95% CI {c.borehole_ci}), permutation p = {c.borehole_p:.3f}.",
             f"- **Suggested field-check locations** (points; their exact position is not significant): "
             f"{h.loc['top 10', 'hits_within_100m']} of the top 10 lie within 100 m of a lode "
             f"(expected at random: {h.loc['top 10', 'expected_random_100m']}); "
             f"{h.loc['all', 'hits_within_100m']} of all {h.loc['all', 'n_targets']} "
             f"(expected {h.loc['all', 'expected_random_100m']}). The method predicts regions far better than points.\n",
             ]
    if zcap is not None:
        lines += ["**Prospective zones against the known lodes.** Enrichment = share of lode area inside the zones divided "
                  "by the share of the block they cover (1.0 = no better than chance).\n",
                  "| Zones | % of block | % of lode area inside | Enrichment | Lode-intersecting boreholes inside | Halo-only boreholes inside |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r.zones} | {r.pct_of_block} | {r.pct_of_truth_captured} | {r.enrichment}× | {r.lode_boreholes_inside} | "
                  f"{r.halo_boreholes_inside} |" for r in zcap.itertuples()]
        lines += [""]
    lines += ["| Layer | Lode AUC (95% CI) | Borehole AUC (95% CI) |", "|---|---|---|"]
    lines += [f"| {r.layer} | {r.lode_auc:.2f} ({r.lode_ci}) | {r.borehole_auc:.2f} ({r.borehole_ci}) |"
              for r in res.itertuples()]
    lines += ["\nAUC = probability that a mineralised location scores higher than a non-mineralised one; 0.5 = no skill. "
              "Lode polygons are plan outlines of sub-surface lodes, and boreholes are inclined (~60°), so collars are "
              "offset from where they cut the lode. Neighbouring pixels are correlated, so pixel-level p-values are optimistic."]
    return lines


def kehalpur(out: Path):
    name = "kehalpur"
    aoi, grid, aoi_utm, inside, G, targets, zones, zmask = _setup(name)
    gt = ROOT / "data" / name / "ground_truth"
    lith = gpd.read_file(gt / "lithology_gsi_cro19966.geojson").to_crs(grid.crs)
    lith["group"] = lith["LITHOLOGICAL_UNIT"].map(lith_group)
    cls = np.zeros(grid.shape, "uint8")
    for g in (COVER, NONHOST, HOST):
        sub = lith[lith["group"] == g]
        cls[rasterize([(x, 1) for x in sub.geometry], out_shape=grid.shape, transform=grid.transform,
                      dtype="uint8").astype(bool)] = g
    cls[~inside] = 0
    basement = (cls == HOST) | (cls == NONHOST)
    idx = gpd.read_file(gt / "ngdr_report_index/EXPLORATION_DATA_GIS_VIEW_REPORT_CLIP.shp").to_crs(grid.crs)
    xy = targets[["Easting", "Northing"]].to_numpy()

    rows = []
    for k in LAYERS:
        a = _auc(G[k], basement, cls == COVER, inside)
        rows.append({"layer": LABEL[k], "basement_vs_cover_auc": a["auc"], "ci": f"{a['ci_low']:.2f}–{a['ci_high']:.2f}"})
    lith_res = pd.DataFrame(rows)
    lith_res.to_csv(out / "tables/basement_vs_cover_auc.csv", index=False)

    brow = []
    for rid, label in NGDR.items():
        poly = idx[idx.exp_upid == rid].union_all().intersection(aoi_utm)
        m = rasterize([(poly, 1)], out_shape=grid.shape, transform=grid.transform).astype(bool) & inside
        a = _auc(G["cpi"], m, ~m, inside)
        n_in = int(sum(poly.contains(Point(x, y)) for x, y in xy))
        brow.append({"report": rid, "area": label, "area_km2": round(poly.area / 1e6, 2), "cpi_auc": a["auc"],
                     "ci": f"{a['ci_low']:.2f}–{a['ci_high']:.2f}", "locations_inside": n_in,
                     "locations_expected_at_random": round(len(xy) * poly.area / aoi_utm.area, 1),
                     "priority_zones_touching": int(zones[~zones["Priority_Tier"].str.contains("Tier 3")].intersects(poly).sum())
                     if zones is not None else None})
    blocks = pd.DataFrame(brow)
    blocks.to_csv(out / "tables/ngdr_exploration_blocks.csv", index=False)
    zcap = None
    if zmask:
        key = ["CRO-20859-1986", "CRO-23431-2018", "CRO-22666-2009"]
        explored = rasterize([(g, 1) for g in idx[idx.exp_upid.isin(key)].geometry], out_shape=grid.shape,
                             transform=grid.transform).astype(bool) & inside
        zcap = zone_capture(zmask, explored, inside)
        zcap["pct_of_zone_area_on_cover"] = [round(100 * float((m & (cls == COVER)).sum() / max(m.sum(), 1)), 1)
                                             for m in zmask.values()]
        zcap.to_csv(out / "tables/zone_capture.csv", index=False)

    tc = np.clip(((xy[:, 0] - grid.x0) / grid.res).astype(int), 0, grid.width - 1)
    tr = np.clip(((grid.y0 - xy[:, 1]) / grid.res).astype(int), 0, grid.height - 1)
    on = cls[tr, tc]
    share = {g: (cls == g).sum() / inside.sum() for g in (COVER, NONHOST, HOST)}
    tl = pd.DataFrame([{"lithology_group": n, "locations": int((on == g).sum()),
                        "expected_by_area": round(share[g] * len(xy), 1)}
                       for g, n in ((HOST, "granitic / metapelite"), (NONHOST, "amphibolite / mafic–ultramafic"),
                                    (COVER, "Deccan trap / infra-trap cover"))])
    tl.to_csv(out / "tables/field_check_locations_by_lithology.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=170)
    cf = _panel(ax, grid, G["cpi"], inside, aoi, "Kehalpur — CPI with NGDR exploration blocks and GSI Deccan cover")
    colors = ["#00e676", "#76ff03", "#ffea00", "#00e5ff"]
    for (rid, label), col in zip(NGDR.items(), colors):
        idx[idx.exp_upid == rid].to_crs(4326).boundary.plot(ax=ax, color=col, linewidth=1.6, zorder=6)
        ax.plot([], [], color=col, lw=2, label=f"{rid}: {label}")
    lith[lith["group"] == COVER].to_crs(4326).boundary.plot(ax=ax, color="white", linewidth=1.0, ls="--", zorder=5)
    ax.plot([], [], color="white", lw=1.5, ls="--", label="Deccan trap / infra-trap (GSI)")
    if zones is not None and len(zones):
        zones[~zones["Priority_Tier"].str.contains("Tier 3")].to_crs(4326).boundary.plot(ax=ax, color="#d500f9", linewidth=1.4, zorder=7)
        ax.plot([], [], color="#d500f9", lw=2, label="Prospective zone (Tier 1 / 2)")
    ax.scatter(targets.Longitude[:10], targets.Latitude[:10], s=45, marker="*", c="#ffd600", edgecolors="black",
               linewidths=0.6, zorder=8, label="Suggested field-check locations (top 10)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7.5, framealpha=0.9)
    fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08, shrink=0.6, label="CPI (%)")
    fig.tight_layout()
    fig.savefig(out / "maps/cpi_vs_ngdr_blocks_lithology.png", bbox_inches="tight")
    plt.close(fig)

    c = lith_res.iloc[0]
    lines = ["# Kehalpur — geological-consistency checks\n",
             "**These are not a validation against copper.** No borehole or lode coordinates were available for this "
             "block. The checks below ask whether the results are consistent with independent GSI information.\n",
             f"- **CPI on Precambrian basement vs Deccan / infra-trap cover** (GSI CRO-19966-1987; the cover is younger "
             f"than the mineralisation): AUC {c.basement_vs_cover_auc:.2f} (95% CI {c.ci}). "
             f"{int(tl.loc[2, 'locations'])} of {len(xy)} suggested field-check locations fall on cover "
             f"(expected at random: {tl.loc[2, 'expected_by_area']}).",
             "- **CPI inside NGDR exploration-block footprints vs the rest of the block:**\n",
             "| NGDR report | Area | km² | CPI AUC (95% CI) | Priority zones touching it | Field-check locations inside (random) |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r.report} | {r.area} | {r.area_km2} | {r.cpi_auc:.2f} ({r.ci}) | {r.priority_zones_touching} | "
              f"{r.locations_inside} ({r.locations_expected_at_random}) |" for r in blocks.itertuples()]
    if zcap is not None:
        lines += ["\n- **Prospective zones against the three explored / mineralised NGDR blocks** (Bhawratekra and the two "
                  "Banskhapa-Pipariya areas). Enrichment = share of explored ground inside the zones divided by the share of "
                  "the block they cover (1.0 = no better than chance).\n",
                  "| Zones | % of block | % of explored ground inside | Enrichment | % of zone area on Deccan cover |",
                  "|---|---|---|---|---|"]
        lines += [f"| {r.zones} | {r.pct_of_block} | {r.pct_of_truth_captured} | {r.enrichment}× | {r.pct_of_zone_area_on_cover} |"
                  for r in zcap.itertuples()]
    lines += ["\nNGDR footprints are areas that were explored, not ore outlines, and they were chosen partly because of "
              "surface signs (e.g. gossans) that this method also detects. AUC 0.5 = no difference from the rest of the block."]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=["thanewasna", "kehalpur"])
    ap.add_argument("--output", help="output folder to validate (default: the block name)")
    args = ap.parse_args(argv)
    global OUTPUT
    OUTPUT = args.output
    out = ROOT / "outputs" / f"{args.output or args.name}_validation"
    for d in ("tables", "maps"):
        (out / d).mkdir(parents=True, exist_ok=True)
    lines = thanewasna(out) if args.name == "thanewasna" else kehalpur(out)
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
