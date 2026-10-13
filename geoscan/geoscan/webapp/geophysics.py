"""Reference geophysics layers (view and download only; never used in the CPI or the zone ranking).

Source: GSI Bhukosh ground gravity / magnetic station exports placed in
    data/<area>/ground_truth/bhukosh/GRAVITY_CLIP/   (field bouguer_an, mGal)
    data/<area>/ground_truth/bhukosh/MAGNETIC_CLIP/  (field magnetic_a, nT)

For each layer: the stations (duplicates removed), and - when there are enough stations - a surface
interpolated between them (thin-plate spline), masked where no station is near, with contour lines.
The surface is an interpolation of sparse points (stations are typically ~1 km apart): it shows the
regional pattern only and must not be read at the scale of a single zone.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from matplotlib import colormaps
from PIL import Image
from rasterio.features import rasterize
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from skimage.measure import find_contours

from ..aoi import load_aoi
from ..grid import Grid, grid_for_aoi
from ..raster_io import write_raster
from . import catalog
from .layers import _to_mercator
from rasterio.warp import Resampling

LAYERS = {
    "magnetic": {"folder": "MAGNETIC_CLIP", "field": "magnetic_a", "label": "Magnetic anomaly (ground stations)",
                 "unit": "nT", "cmap": "RdBu_r", "diverging": True,
                 "extra": {"observed_m": "observed_total_field_nT", "igrf_nt": "igrf_nT"},
                 "what": "Ground magnetic anomaly: the measured total field minus the Earth's reference field (IGRF). "
                         "Highs mark magnetite-rich rock (mafic units); lows can mark non-magnetic rock or zones where "
                         "fluids destroyed magnetite."},
    "gravity": {"folder": "GRAVITY_CLIP", "field": "bouguer_an", "label": "Bouguer gravity anomaly (ground stations)",
                "unit": "mGal", "cmap": "viridis", "diverging": False,
                "extra": {"elevation_": "elevation_m", "observed_g": "observed_gravity_mGal",
                          "theoretica": "theoretical_gravity_mGal"},
                "what": "Bouguer gravity anomaly: gravity corrected for latitude, height and terrain mass. "
                        "Highs mark denser rock (mafic units, massive sulphide); lows mark lighter rock (granite, cover)."},
}
SOURCE = "Geological Survey of India, Bhukosh portal (ground gravity / magnetic stations)"
CAVEAT = ("Reference only: not used in the CPI or the zone ranking. The surface and contours are interpolated "
          "between sparse stations and show the regional pattern only.")
MIN_STATIONS_FOR_SURFACE = 6
GRID_RES_M = 60.0


def _dir(block: str, key: str) -> Path:
    return catalog.DATA / block / "ground_truth" / "bhukosh" / LAYERS[key]["folder"]


def _cache(block: str) -> Path:
    p = catalog.OUTPUTS / "_reference_cache" / block
    p.mkdir(parents=True, exist_ok=True)
    return p


def _aoi(block: str):
    aoi = load_aoi(catalog.DATA / block / "aoi" / "aoi.geojson", block)
    return aoi, grid_for_aoi(aoi, GRID_RES_M, 180.0)


def stations(block: str, key: str) -> gpd.GeoDataFrame | None:
    """Stations in the area's UTM zone, exact duplicates (within 10 m) removed, tidy column names."""
    d = _dir(block, key)
    if not d.exists() or not any(d.glob("*.shp")):
        return None
    spec = LAYERS[key]
    aoi, grid = _aoi(block)
    g = gpd.read_file(d).to_crs(grid.crs)
    g = g[np.isfinite(g[spec["field"]].astype(float))].reset_index(drop=True)
    if len(g) == 0:
        return None
    xy = np.c_[g.geometry.x, g.geometry.y]
    tree, keep = cKDTree(xy), np.ones(len(g), bool)
    for i in range(len(g)):
        if keep[i]:
            for j in tree.query_ball_point(xy[i], 10.0):
                if j > i:
                    keep[j] = False
    g = g[keep].reset_index(drop=True)
    out = gpd.GeoDataFrame({"station": np.arange(1, len(g) + 1), "value": g[spec["field"]].astype(float).round(3),
                            "unit": spec["unit"]}, geometry=g.geometry, crs=grid.crs)
    for src, name in spec["extra"].items():
        if src in g:
            out[name] = g[src].astype(float).round(3)
    ll = out.geometry.to_crs(4326)
    out["longitude"], out["latitude"] = ll.x.round(6), ll.y.round(6)
    out["inside_area"] = out.within(aoi.to_crs(grid.crs))
    return out


def _spacing(g: gpd.GeoDataFrame) -> float:
    xy = np.c_[g.geometry.x, g.geometry.y]
    if len(xy) < 2:
        return float("nan")
    return float(np.median(cKDTree(xy).query(xy, 2)[0][:, 1]))


def surface(block: str, key: str):
    """Interpolated surface on a 60 m grid (NaN outside the area or far from any station), or (None, grid)."""
    g = stations(block, key)
    aoi, grid = _aoi(block)
    if g is None or len(g) < MIN_STATIONS_FOR_SURFACE:
        return None, grid
    xy = np.c_[g.geometry.x, g.geometry.y]
    xs = grid.x0 + (np.arange(grid.width) + 0.5) * grid.res
    ys = grid.y0 - (np.arange(grid.height) + 0.5) * grid.res
    X, Y = np.meshgrid(xs, ys)
    pts = np.c_[X.ravel(), Y.ravel()]
    centre, scale = xy.mean(0), float(xy.std(0).mean())      # spread of the stations (east and north separately)
    rbf = RBFInterpolator((xy - centre) / scale, g["value"].values, kernel="thin_plate_spline", smoothing=1e-3)
    z = rbf((pts - centre) / scale).reshape(grid.shape)
    near = cKDTree(xy).query(pts)[0].reshape(grid.shape) <= 1.5 * _spacing(g)
    inside = rasterize([(aoi.to_crs(grid.crs), 1)], out_shape=grid.shape, transform=grid.transform).astype(bool)
    lo, hi = g["value"].min(), g["value"].max()
    z = np.clip(z, lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))      # no wild overshoot between stations
    return np.where(near & inside, z, np.nan).astype("float32"), grid


def _colour_range(g: gpd.GeoDataFrame) -> tuple[float, float]:
    """Colour limits: the 2nd-98th percentile of the station values (relative lows to relative highs)."""
    lo, hi = np.percentile(g["value"], [2, 98]) if len(g) >= 10 else (g["value"].min(), g["value"].max())
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    return float(lo), float(hi)


def _nice_step(span: float, target: int = 9) -> float:
    raw = span / max(target, 1)
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
    return float(min((1, 2, 2.5, 5, 10), key=lambda m: abs(m * mag - raw)) * mag)


def _levels(z: np.ndarray) -> list[float]:
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    if hi <= lo:
        return []
    step = _nice_step(hi - lo)
    return [round(v, 6) for v in np.arange(np.ceil(lo / step) * step, hi, step)]


def contours(block: str, key: str) -> gpd.GeoDataFrame | None:
    z, grid = surface(block, key)
    if z is None:
        return None
    filled = np.where(np.isfinite(z), z, np.nanmin(z) - 1e6)      # contours stop at the masked edge
    recs, geoms = [], []
    for lev in _levels(z):
        lines = []
        for path in find_contours(filled, lev):
            ok = np.isfinite(z[np.clip(path[:, 0].round().astype(int), 0, grid.height - 1),
                               np.clip(path[:, 1].round().astype(int), 0, grid.width - 1)])
            xs = grid.x0 + (path[:, 1] + 0.5) * grid.res
            ys = grid.y0 - (path[:, 0] + 0.5) * grid.res
            seg = []
            for x, y, k in zip(xs, ys, ok):
                if k:
                    seg.append((x, y))
                else:
                    if len(seg) > 2:
                        lines.append(LineString(seg))
                    seg = []
            if len(seg) > 2:
                lines.append(LineString(seg))
        if lines:
            m = unary_union(lines).simplify(grid.res / 3)
            geoms.append(m if isinstance(m, (LineString, MultiLineString)) else MultiLineString(
                [x for x in getattr(m, "geoms", []) if isinstance(x, LineString)]))
            recs.append({"layer": key, "level": lev, "unit": LAYERS[key]["unit"]})
    return gpd.GeoDataFrame(recs, geometry=geoms, crs=grid.crs) if recs else None


def info(block: str, include_missing: bool = False) -> list[dict]:
    """Layers for an area, with the facts a reader needs to judge them.

    include_missing=True also returns layers with no data ('available': False and the reason), so the
    app can list the same options for every area and say why one is empty.
    """
    out = []
    for key, spec in LAYERS.items():
        g = stations(block, key)
        if g is None:
            if include_missing:
                out.append({"key": key, "kind": "stations", "label": spec["label"], "unit": spec["unit"],
                            "available": False, "what": spec["what"], "source": SOURCE, "caveat": CAVEAT,
                            "reason": "No ground gravity / magnetic stations have been added for this area."})
            continue
        sp = _spacing(g)
        out.append({"key": key, "kind": "stations", "available": True, "label": spec["label"], "unit": spec["unit"],
                    "stations": int(len(g)),
                    "stations_inside": int(g["inside_area"].sum()),
                    "spacing_m": None if np.isnan(sp) else round(sp),
                    "min": float(g["value"].min()), "max": float(g["value"].max()),
                    "has_surface": len(g) >= MIN_STATIONS_FOR_SURFACE, "what": spec["what"],
                    "source": SOURCE, "caveat": CAVEAT})
    return out


def overlay(block: str, key: str) -> dict:
    """Coloured surface PNG (EPSG:3857) + bounds + legend for the web map; {'none': True} without a surface."""
    c = _cache(block)
    png, meta = c / f"{key}.png", c / f"{key}.json"
    src = next(_dir(block, key).glob("*.shp"))
    if png.exists() and meta.exists() and png.stat().st_mtime >= src.stat().st_mtime:
        return json.loads(meta.read_text())
    spec = LAYERS[key]
    z, grid = surface(block, key)
    g = stations(block, key)
    lo, hi = _colour_range(g)
    stops = [colormaps[spec["cmap"]](t) for t in np.linspace(0, 1, 9)]
    legend = {"label": spec["label"], "unit": spec["unit"], "min": round(lo, 1), "max": round(hi, 1), "kind": "ramp",
              "colors": ["#%02x%02x%02x" % tuple(int(255 * v) for v in s[:3]) for s in stops],
              "levels": [], "reference": True}
    if z is None:
        info_ = {"none": True, "bounds": None, "legend": legend}
        png.write_bytes(b"")
    else:
        arr, bounds = _to_mercator(z, grid, 4, Resampling.bilinear)
        x = np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)
        rgba = (colormaps[spec["cmap"]](np.nan_to_num(x)) * 255).astype("uint8")
        rgba[..., 3] = np.where(np.isfinite(arr), 205, 0)
        Image.fromarray(rgba, "RGBA").save(png)
        legend["contour_step"] = _nice_step(float(np.nanmax(z) - np.nanmin(z)))
        info_ = {"bounds": bounds, "legend": legend}
    meta.write_text(json.dumps(info_))
    return info_


def figure(block: str, key: str) -> Path:
    """Publication-style PNG: surface, labelled contours and stations with their values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from pyproj import Transformer
    out = _cache(block) / f"reference_{key}_map.png"
    src = next(_dir(block, key).glob("*.shp"))
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    spec = LAYERS[key]
    g = stations(block, key)
    z, grid = surface(block, key)
    aoi, _ = _aoi(block)
    ov = overlay(block, key)["legend"]
    fig, ax = plt.subplots(figsize=(11, 8), dpi=170)
    if z is not None:
        ll = Transformer.from_crs(grid.crs, 4326, always_xy=True)
        xs = grid.x0 + (np.arange(grid.width) + 0.5) * grid.res
        ys = grid.y0 - (np.arange(grid.height) + 0.5) * grid.res
        LON, LAT = ll.transform(*np.meshgrid(xs, ys))
        cf = ax.contourf(LON, LAT, z, levels=np.linspace(ov["min"], ov["max"], 41), cmap=spec["cmap"], extend="both")
        lv = _levels(z)
        if lv:
            cs = ax.contour(LON, LAT, z, levels=lv, colors="black", linewidths=0.6)
            ax.clabel(cs, fmt="%g", fontsize=7)
        fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08, shrink=0.6, ticks=MaxNLocator(8),
                     label=f"{spec['label']} ({spec['unit']})")
    sc = ax.scatter(g["longitude"], g["latitude"], c=g["value"], cmap=spec["cmap"], vmin=ov["min"], vmax=ov["max"],
                    s=46, edgecolors="black", linewidths=0.8, zorder=5)
    if z is None:
        fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, shrink=0.6, ticks=MaxNLocator(8),
                     label=f"{spec['label']} ({spec['unit']})")
    span = float(g["value"].max() - g["value"].min())
    for _, r in g.iterrows():
        ax.annotate(f"{r['value']:.0f}" if span >= 50 else f"{r['value']:g}", (r["longitude"], r["latitude"]), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    x, y = aoi.geometry.exterior.xy
    ax.plot(x, y, color="black", lw=1.6)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(y))))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    sp = _spacing(g)
    ax.set_title(f"{block.capitalize()}: {spec['label']}\n"
                 f"{len(g)} stations" + ("" if np.isnan(sp) else f", about {sp / 1000:.1f} km apart") +
                 (" · surface interpolated between stations" if z is not None else " · too few stations for a surface") +
                 "\nReference only: not used in the CPI or the zone ranking", fontsize=10, fontweight="bold")
    fig.text(0.01, 0.01, f"Source: {SOURCE}", fontsize=6.5, color="#555")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def raster_file(block: str, key: str, path: Path) -> bool:
    """Write the interpolated surface as a GeoTIFF (UTM). Returns False when there is no surface."""
    z, grid = surface(block, key)
    if z is None:
        return False
    g = stations(block, key)
    write_raster(path, z[None], grid, [f"{key}_{LAYERS[key]['unit']}"],
                 tags={"source": SOURCE, "stations": len(g), "median_station_spacing_m": round(_spacing(g)),
                       "method": "thin-plate-spline interpolation of ground stations; masked beyond 1.5 x spacing",
                       "note": CAVEAT})
    return True
