"""GeoTIFF read/write bound to a :class:`Grid` (shape and orientation are enforced)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.features import geometry_mask

from .grid import Grid


def write_raster(path: str | Path, data: np.ndarray, grid: Grid, band_names: Sequence[str] | None = None,
                 nodata: float | None = np.nan, dtype: str = "float32", tags: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = data[None] if data.ndim == 2 else data
    if arr.shape[1:] != grid.shape:
        raise ValueError(f"{path.name}: array shape {arr.shape[1:]} does not match grid {grid.shape}")
    if band_names is not None and len(band_names) != arr.shape[0]:
        raise ValueError(f"{path.name}: {len(band_names)} band names for {arr.shape[0]} bands")

    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=arr.shape[0], dtype=dtype,
                   crs=grid.crs, transform=grid.transform, nodata=nodata, compress="deflate",
                   predictor=2 if dtype.startswith("int") or dtype.startswith("uint") else 3,
                   tiled=grid.width >= 256 and grid.height >= 256, BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype))
        if band_names:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
        if tags:
            dst.update_tags(**{k: (v if isinstance(v, str) else json.dumps(v)) for k, v in tags.items()})
    return path


def read_raster(path: str | Path) -> tuple[np.ndarray, Grid, list[str], dict]:
    with rasterio.open(path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0 or t.e >= 0 or abs(t.a + t.e) > 1e-9:
            raise ValueError(f"{path}: not a north-up square-pixel raster")
        grid = Grid(src.crs.to_string(), t.c, t.f, t.a, src.width, src.height)
        data = src.read().astype("float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            data[data == src.nodata] = np.nan
        names = [d or f"band_{i + 1}" for i, d in enumerate(src.descriptions)]
        tags = src.tags()
    return data, grid, names, tags


def read_band(path: str | Path, name: str) -> np.ndarray:
    data, _, names, _ = read_raster(path)
    return data[names.index(name)]


def aoi_mask(grid: Grid, geometry_in_grid_crs) -> np.ndarray:
    """True inside the AOI polygon."""
    return geometry_mask([geometry_in_grid_crs], out_shape=grid.shape, transform=grid.transform, invert=True)
