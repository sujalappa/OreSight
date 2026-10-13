"""Reference geochemistry: stream sediment samples (view and download only; never used in the CPI).

Source: GSI Bhukosh stream sediment export placed in
    data/<area>/ground_truth/bhukosh/STREAM_SEDIMENTS_CLIP/

Shown as points only. A stream sediment sample describes the ground upstream of where it was taken (its
catchment), not the sampling spot, so no surface is interpolated between samples.

Records whose values are clearly in the wrong columns are dropped, and the layer is reported as not
available when fewer than MIN_SAMPLES usable records remain (so the app can say why).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np

from ..aoi import load_aoi
from ..grid import grid_for_aoi
from . import catalog

KEY = "stream_sediment"
LABEL = "Stream sediment geochemistry (samples)"
FOLDER = "STREAM_SEDIMENTS_CLIP"
SOURCE = "Geological Survey of India, Bhukosh portal (stream sediment samples)"
MIN_SAMPLES = 5
WHAT = ("Chemical analyses of fine sediment collected from stream beds. Each sample reflects the rocks and soils "
        "of the ground that drains to that point, so a high value points upstream, not at the sampling spot.")
CAVEAT = ("Reference only: not used in the CPI or the zone ranking. Shown as points, with no surface between them, "
          "because each sample stands for its upstream catchment. High values can simply reflect the rock type: "
          "basalt and other mafic rocks are naturally rich in Cu, Ni, Cr, Co and V. Units are not stated in the source "
          "file (trace elements are normally ppm, gold ppb, oxides weight %).")
OXIDES = {"sio2": "SiO₂", "al2o3": "Al₂O₃", "fe2o3": "Fe₂O₃", "tio2": "TiO₂", "cao": "CaO", "mgo": "MgO", "mno": "MnO",
          "na2o": "Na₂O", "k2o": "K₂O", "p2o5": "P₂O₅", "loi": "LOI"}
FIRST = ["cu", "zn", "pb", "au", "ag", "as_", "ni", "co", "cr", "mo", "sb", "bi", "ba", "v"]       # shown first
SKIP = {"gid", "objectid", "sampleno", "longitude", "latitude", "toposheet", "geometry"}
MAP_ELEMENTS = ("cu", "zn", "pb", "au")                                                         # maps in a download


def element_label(col: str) -> str:
    return OXIDES.get(col) or col.rstrip("_").capitalize()


def _dir(block: str) -> Path:
    return catalog.DATA / block / "ground_truth" / "bhukosh" / FOLDER


def _load(block: str):
    """(usable samples in UTM or None, total records in the file, reason when unusable)."""
    d = _dir(block)
    if not d.exists() or not any(d.glob("*.shp")):
        return None, 0, "No stream sediment data has been added for this area."
    aoi = load_aoi(catalog.DATA / block / "aoi" / "aoi.geojson", block)
    grid = grid_for_aoi(aoi, 60.0, 180.0)
    g = gpd.read_file(d).to_crs(grid.crs)
    total = len(g)
    # values in the wrong columns: gallium / scandium far outside any natural range, or no loss-on-ignition at all
    bad = np.zeros(total, bool)
    for col, limit in (("ga", 60.0), ("sc", 80.0)):
        if col in g:
            bad |= g[col].astype(float) > limit
    if "loi" in g:
        bad |= g["loi"].astype(float) <= 0
    g = g[~bad].reset_index(drop=True)
    if len(g) < MIN_SAMPLES:
        n_bad = int(bad.sum())
        why = (f"Only {total} sample{'s' if total != 1 else ''} fall{'s' if total == 1 else ''} in this area"
               + ("" if not n_bad else ", and its values are in the wrong columns" if total == 1
                  else f", and {n_bad} of them have values in the wrong columns")
               + f". At least {MIN_SAMPLES} usable samples are needed to show a meaningful layer.")
        return None, total, why
    return g, total, ""


def elements(g: gpd.GeoDataFrame) -> list[str]:
    cols = []
    for c in g.columns:
        if c in SKIP or not np.issubdtype(g[c].dtype, np.number):
            continue
        v = g[c].astype(float)
        if v.std() == 0 or (v == 0).all():            # below detection everywhere, or not analysed
            continue
        cols.append(c)
    return [c for c in FIRST if c in cols] + sorted(c for c in cols if c not in FIRST and c not in OXIDES) + \
        [c for c in OXIDES if c in cols]


def samples(block: str) -> gpd.GeoDataFrame | None:
    """Usable samples (UTM) with a tidy id, lon/lat and every analysed element."""
    g, _, _ = _load(block)
    if g is None:
        return None
    cols = elements(g)
    out = gpd.GeoDataFrame({"sample": g["sampleno"].astype(str) if "sampleno" in g else np.arange(1, len(g) + 1)},
                           geometry=g.geometry, crs=g.crs)
    ll = out.geometry.to_crs(4326)
    out["longitude"], out["latitude"] = ll.x.round(6), ll.y.round(6)
    for c in cols:
        out[c.rstrip("_")] = g[c].astype(float).round(4)
    return out


def _rock_note(block: str, g: gpd.GeoDataFrame) -> str:
    """Copper by mapped rock unit, if a geology map is available (guards against reading basalt as ore).

    States only what the map supports: samples outside the mapped geology are reported as not mapped.
    """
    if "cu" not in g:
        return ""
    lith = next(iter(sorted((catalog.DATA / block / "ground_truth").glob("lithology*.geojson"))), None)
    if lith is None:
        return ""
    try:
        units = gpd.read_file(lith).to_crs(g.crs)
        field = next(c for c in ("LITHOLOGICAL_UNIT", "LITHOLOGY", "UNIT") if c in units)
        is_mafic = units[field].astype(str).str.contains("Deccan|basalt|trap", case=False, na=False) & \
            ~units[field].astype(str).str.contains("Infra|meta", case=False, na=False)
        cover, mapped = units[is_mafic].union_all(), units.union_all()
        cu = g["cu"].astype(float)
        on_cover, on_map = g.within(cover), g.within(mapped)
        other = on_map & ~on_cover
        if on_cover.sum() < 2 or other.sum() < 2:
            return ""
        note = (f"Copper by mapped rock unit in this area: median {cu[on_cover].median():.0f} in the {int(on_cover.sum())} "
                f"samples on Deccan basalt, against {cu[other].median():.0f} in the {int(other.sum())} samples on the older "
                "basement rocks. Basalt is naturally copper-rich and younger than the mineralisation, so highs on basalt "
                "reflect the rock type, not a deposit.")
        top = cu >= cu.quantile(0.75)
        unmapped = int((top & ~on_map).sum())
        if unmapped:
            note += (f" {unmapped} of the {int(top.sum())} highest-copper samples lie outside the geology map used here, "
                     "so their rock unit is not confirmed.")
        return note
    except Exception:
        return ""


def info(block: str) -> dict:
    """Layer description for the app. 'available' False comes with the reason, so the app can show it."""
    g, total, why = _load(block)
    base = {"key": KEY, "kind": "samples", "label": LABEL, "source": SOURCE, "what": WHAT, "caveat": CAVEAT}
    if g is None:
        return {**base, "available": False, "reason": why, "records": total}
    xy = np.c_[g.geometry.x, g.geometry.y]
    from scipy.spatial import cKDTree
    spacing = float(np.median(cKDTree(xy).query(xy, 2)[0][:, 1])) if len(xy) > 1 else None
    els = []
    for c in elements(g):
        v = g[c].astype(float)
        els.append({"key": c.rstrip("_"), "label": element_label(c), "min": round(float(v.min()), 3),
                    "median": round(float(v.median()), 3), "max": round(float(v.max()), 3)})
    return {**base, "available": True, "samples": int(len(g)), "records": total, "dropped": total - len(g),
            "spacing_m": round(spacing) if spacing else None, "elements": els, "note": _rock_note(block, g)}


def figure(block: str, element: str) -> Path | None:
    """PNG: samples coloured and sized by one element, each labelled with its value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = samples(block)
    if g is None or element not in g:
        return None
    out = catalog.OUTPUTS / "_reference_cache" / block
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"reference_stream_sediment_{element}.png"
    src = next(_dir(block).glob("*.shp"))
    if path.exists() and path.stat().st_mtime >= src.stat().st_mtime:
        return path
    aoi = load_aoi(catalog.DATA / block / "aoi" / "aoi.geojson", block)
    v = g[element].astype(float)
    lo, hi = float(v.min()), float(v.max())
    size = 40 + 220 * (v - lo) / (hi - lo if hi > lo else 1.0)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=170)
    sc = ax.scatter(g["longitude"], g["latitude"], c=v, s=size, cmap="YlOrRd", vmin=lo, vmax=hi, edgecolors="black",
                    linewidths=0.8, zorder=5)
    for _, r in g.iterrows():
        ax.annotate(f"{r[element]:g}", (r["longitude"], r["latitude"]), xytext=(6, 6), textcoords="offset points", fontsize=7)
    x, y = aoi.geometry.exterior.xy
    ax.plot(x, y, color="black", lw=1.6)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(y))))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    lab = element_label(element if element != "as" else "as_")
    fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, shrink=0.6, label=f"{lab} in stream sediment (units not stated in source)")
    ax.set_title(f"{block.capitalize()}: {lab} in stream sediment samples ({len(g)} samples)\n"
                 "Each sample reflects the ground upstream of it; no surface is drawn between samples\n"
                 "Reference only: not used in the CPI or the zone ranking", fontsize=10, fontweight="bold")
    note = _rock_note(block, gpd.GeoDataFrame({"cu": g["cu"]}, geometry=g.geometry, crs=g.crs)) if element == "cu" and "cu" in g else ""
    fig.text(0.01, 0.01, (note + "\n" if note else "") + f"Source: {SOURCE}", fontsize=6.5, color="#555", wrap=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
