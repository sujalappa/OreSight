"""Map layers for the web viewer: coloured raster overlays, contour lines and vectors."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from matplotlib import colormaps
from PIL import Image
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from scipy.ndimage import zoom
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from skimage.measure import find_contours

from ..aoi import load_aoi
from ..prospectivity.alteration import CONTOUR_LEVELS, LABELS
from ..raster_io import read_raster

STACK = "rasters/alteration/alteration_stack_30m.tif"
RASTER_LAYERS = {
    "cpi": {"file": "rasters/targets/cpi.tif", "band": "cpi", "label": "Copper Prospectivity Index (CPI)",
            "levels": (28, 50, 70), "unit": "%", "scale": 100.0, "cmap": "plasma", "range": (0, 100)},
    **{m: {"file": STACK, "band": m, "label": LABELS[m], "levels": CONTOUR_LEVELS[m], "unit": "%",
           "scale": 100.0, "cmap": "plasma", "range": (0, 100)} for m in ("muscovite", "chlorite", "gossan", "quartz")},
    "lineament_density": {"file": "rasters/structure/lineament_density.tif", "band": "density_km_per_km2",
                          "label": "Lineament density", "levels": None, "unit": "km/km²", "scale": 1.0,
                          "cmap": "viridis", "range": None},
}
DROP_COLUMNS = {"EMIT_Data_Support"}
VECTORS = {"targets": "field_check_locations.geojson", "zones": "prospective_zones.geojson",
           "lineaments": "lineaments.geojson", "cpi_zones": "cpi_target_zones.geojson", "aoi": "aoi.geojson"}
OLD_NAMES = {"field_check_locations.geojson": "top50_targets.geojson"}      # results made before zones existed


def _cache(run_dir: Path) -> Path:
    p = run_dir / "_webapp_cache"
    p.mkdir(exist_ok=True)
    return p


def _fresh(target: Path, *sources: Path) -> bool:
    return target.exists() and all(target.stat().st_mtime >= s.stat().st_mtime for s in sources if s.exists())


def layer_array(run_dir: Path, key: str):
    spec = RASTER_LAYERS[key]
    data, grid, names, _ = read_raster(run_dir / spec["file"])
    return data[names.index(spec["band"])] * spec["scale"], grid


EMIT_LAYERS = ("muscovite", "chlorite", "gossan")


def not_assessed(run_dir: Path, layer: str, min_support: float = 0.20):
    """Same rule as the contour maps: few usable EMIT pixels AND nothing detected for this layer.

    Mineral layers use their own intensity, CPI uses the three EMIT layers together. Quartz (ASTER) and
    lineament density do not depend on EMIT, so they have no 'not assessed' area (returns None).
    """
    if layer not in EMIT_LAYERS and layer != "cpi":
        return None, None
    data, grid, names, _ = read_raster(run_dir / STACK)
    sup = data[names.index("data_support")]
    keys = EMIT_LAYERS if layer == "cpi" else (layer,)
    signal = np.max(np.stack([np.nan_to_num(data[names.index(k)]) for k in keys]), 0)
    return np.isfinite(sup) & (sup < min_support) & (signal < 0.05), grid


def _to_mercator(arr, grid, upsample: int, resampling):
    dst_crs = "EPSG:3857"
    w, s, e, n = grid.bounds
    tr, W, H = calculate_default_transform(grid.crs, dst_crs, grid.width, grid.height, w, s, e, n,
                                           resolution=None)
    tr = tr * tr.scale(1 / upsample, 1 / upsample)
    W, H = W * upsample, H * upsample
    out = np.full((H, W), np.nan, "float32")
    reproject(arr.astype("float32"), out, src_transform=grid.transform, src_crs=grid.crs, dst_transform=tr,
              dst_crs=dst_crs, resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", *array_bounds(H, W, tr))
    return out, [[south, west], [north, east]]


def overlay(run_dir: Path, key: str) -> dict:
    """PNG (EPSG:3857, transparent outside data) + lat/lon bounds + legend for Leaflet."""
    c = _cache(run_dir)
    png, meta = c / f"{key}.png", c / f"{key}.json"
    hatch_for = key[3:] if key.startswith("na_") else None      # "na_<layer>": not-assessed hatch for that layer
    src = run_dir / (STACK if hatch_for else RASTER_LAYERS[key]["file"])
    if _fresh(png, src) and meta.exists():
        return json.loads(meta.read_text())
    if hatch_for:
        mask, grid = not_assessed(run_dir, hatch_for)
        if mask is None or not mask.any():
            info = {"bounds": None, "legend": None, "none": True}
            meta.write_text(json.dumps(info))
            png.write_bytes(b"")
            return info
        arr, bounds = _to_mercator(np.where(mask, 1.0, np.nan), grid, 4, Resampling.nearest)
        rgba = np.zeros(arr.shape + (4,), "uint8")
        yy, xx = np.indices(arr.shape)
        hatch = ((xx + yy) % 10) < 2
        on = np.isfinite(arr)
        rgba[on] = (90, 90, 90, 70)
        rgba[on & hatch] = (40, 40, 40, 200)
        legend = {"label": "Not assessed (no usable EMIT data)", "kind": "hatch"}
    else:
        spec = RASTER_LAYERS[key]
        a, grid = layer_array(run_dir, key)
        arr, bounds = _to_mercator(a, grid, 4, Resampling.bilinear)
        lo, hi = spec["range"] or tuple(np.nanpercentile(a, [2, 98]).round(3))
        x = np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)
        rgba = (colormaps[spec["cmap"]](np.nan_to_num(x)) * 255).astype("uint8")
        rgba[..., 3] = np.where(np.isfinite(arr), 215, 0)
        stops = [colormaps[spec["cmap"]](t) for t in np.linspace(0, 1, 9)]
        legend = {"label": spec["label"], "unit": spec["unit"], "min": float(lo), "max": float(hi), "kind": "ramp",
                  "colors": ["#%02x%02x%02x" % tuple(int(255 * v) for v in s[:3]) for s in stops],
                  "levels": list(spec["levels"]) if spec["levels"] else []}
    Image.fromarray(rgba, "RGBA").save(png)
    info = {"bounds": bounds, "legend": legend}
    meta.write_text(json.dumps(info))
    return info


def contours(run_dir: Path, key: str) -> gpd.GeoDataFrame:
    """Contour lines at the map levels (UTM), clipped to the AOI, smoothed by 3x bicubic upsampling."""
    spec = RASTER_LAYERS[key]
    c = _cache(run_dir) / f"contours_{key}.gpkg"
    src = run_dir / spec["file"]
    if _fresh(c, src):
        return gpd.read_file(c)
    a, grid = layer_array(run_dir, key)
    aoi = load_aoi(run_dir / "vectors" / "aoi.geojson").to_crs(grid.crs)
    f = 3
    up = zoom(np.nan_to_num(a, nan=0.0), f, order=3)
    res = grid.res / f
    recs, geoms = [], []
    for lev in spec["levels"] or []:
        lines = []
        for path in find_contours(up, lev):
            if len(path) < 3:
                continue
            xs = grid.x0 + (path[:, 1] + 0.5) * res
            ys = grid.y0 - (path[:, 0] + 0.5) * res
            lines.append(LineString(np.c_[xs, ys]))
        if not lines:
            continue
        g = unary_union(lines).intersection(aoi).simplify(res / 3)
        if g.is_empty:
            continue
        geoms.append(g if isinstance(g, (LineString, MultiLineString)) else MultiLineString(
            [x for x in getattr(g, "geoms", []) if isinstance(x, LineString)]))
        recs.append({"layer": key, "label": spec["label"], "level": lev, "unit": spec["unit"]})
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs=grid.crs)
    gdf.to_file(c, driver="GPKG")
    return gdf


def vector(run_dir: Path, key: str) -> gpd.GeoDataFrame:
    name = VECTORS.get(key) or f"{key}.geojson"
    path = run_dir / "vectors" / name
    if not path.exists() and name in OLD_NAMES:
        path = run_dir / "vectors" / OLD_NAMES[name]
    g = gpd.read_file(path)
    return g.drop(columns=[c for c in g.columns if c in DROP_COLUMNS])


def geojson(gdf: gpd.GeoDataFrame) -> str:
    return gdf.to_crs(4326).to_json(drop_id=True)
