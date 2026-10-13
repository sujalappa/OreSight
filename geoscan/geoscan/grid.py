"""Analysis grids.

One rule everywhere: arrays are (row, col), row 0 is the NORTH edge, and the affine
transform is ``Affine(res, 0, x0, 0, -res, y0)``. Every raster written by geoscan
carries this transform and is checked against it on write, so layers cannot be
flipped or offset relative to each other.

All grids for one AOI share an origin snapped to ``align_m``; the 30 m terrain grid,
60 m EMIT grid and 90 m ASTER grid therefore have coincident cell edges.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from affine import Affine

from .aoi import AOI


@dataclass(frozen=True)
class Grid:
    crs: str
    x0: float   # west edge
    y0: float   # north edge
    res: float
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0.0, self.x0, 0.0, -self.res, self.y0)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in grid CRS."""
        return (self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0)

    def coarsen(self, factor: int) -> "Grid":
        return Grid(self.crs, self.x0, self.y0, self.res * factor,
                    math.ceil(self.width / factor), math.ceil(self.height / factor))

    def pad(self, metres: float) -> "Grid":
        n = math.ceil(metres / self.res)
        return Grid(self.crs, self.x0 - n * self.res, self.y0 + n * self.res, self.res,
                    self.width + 2 * n, self.height + 2 * n)

    def window_in(self, outer: "Grid") -> tuple[slice, slice]:
        """Row/col slices of this grid inside a larger, aligned grid of the same resolution."""
        if outer.res != self.res or outer.crs != self.crs:
            raise ValueError("Grids must share CRS and resolution")
        c0 = (self.x0 - outer.x0) / self.res
        r0 = (outer.y0 - self.y0) / self.res
        if not (float(c0).is_integer() and float(r0).is_integer()):
            raise ValueError("Grids are not aligned")
        c0, r0 = int(c0), int(r0)
        return slice(r0, r0 + self.height), slice(c0, c0 + self.width)

    def subgrid(self, row0: int, col0: int, height: int, width: int) -> "Grid":
        return Grid(self.crs, self.x0 + col0 * self.res, self.y0 - row0 * self.res, self.res, width, height)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + (np.arange(self.width) + 0.5) * self.res
        ys = self.y0 - (np.arange(self.height) + 0.5) * self.res
        return xs, ys

    def ee_grid(self) -> dict:
        """Grid description for Earth Engine ``computePixels``."""
        return {
            "dimensions": {"width": self.width, "height": self.height},
            "affineTransform": {"scaleX": self.res, "shearX": 0, "translateX": self.x0,
                                "shearY": 0, "scaleY": -self.res, "translateY": self.y0},
            "crsCode": self.crs,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def grid_for_aoi(aoi: AOI, res: float, align_m: float) -> Grid:
    """Smallest aligned grid covering the AOI in its UTM zone."""
    crs = aoi.utm_crs
    minx, miny, maxx, maxy = aoi.to_crs(crs).bounds
    x0 = math.floor(minx / align_m) * align_m
    y0 = math.ceil(maxy / align_m) * align_m
    x1 = math.ceil(maxx / align_m) * align_m
    y1 = math.floor(miny / align_m) * align_m
    return Grid(crs.to_string(), x0, y0, res, int(round((x1 - x0) / res)), int(round((y0 - y1) / res)))
