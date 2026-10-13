"""Area of interest: load KML / KMZ / GeoJSON / Shapefile into one polygon."""

from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import transform as shp_transform, unary_union


@dataclass
class AOI:
    name: str
    geometry: Polygon | MultiPolygon  # EPSG:4326, lon/lat
    source: str

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.geometry.bounds

    @property
    def utm_crs(self) -> CRS:
        c = self.geometry.centroid
        zone = int((c.x + 180) // 6) + 1
        return CRS.from_epsg((32600 if c.y >= 0 else 32700) + zone)

    def to_crs(self, crs: CRS | str):
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        return shp_transform(tr.transform, self.geometry)

    def area_km2(self) -> float:
        return self.to_crs(self.utm_crs).area / 1e6

    def to_geojson(self, path: str | Path) -> None:
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"name": self.name}, "geometry": mapping(self.geometry)}]}
        Path(path).write_text(json.dumps(fc), encoding="utf-8")


def _parse_coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in text.split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def _polygons_from_kml(xml_text: str) -> list[Polygon]:
    # Namespace-agnostic: KML from Google Earth, QGIS and ArcGIS use different prefixes.
    xml_text = re.sub(r'\sxmlns(:\w+)?="[^"]+"', "", xml_text)
    xml_text = re.sub(r"<(/?)\w+:", r"<\1", xml_text)
    xml_text = re.sub(r'\s\w+:\w+="[^"]*"', "", xml_text)
    root = ET.fromstring(xml_text)
    polys = []
    for poly in root.iter():
        if not poly.tag.split("}")[-1] == "Polygon":
            continue
        outer, holes = None, []
        for el in poly.iter():
            tag = el.tag.split("}")[-1]
            if tag in ("outerBoundaryIs", "innerBoundaryIs"):
                coords = next((c for c in el.iter() if c.tag.split("}")[-1] == "coordinates"), None)
                if coords is None or not coords.text:
                    continue
                ring = _parse_coords(coords.text)
                if tag == "outerBoundaryIs":
                    outer = ring
                else:
                    holes.append(ring)
        if outer and len(outer) >= 3:
            polys.append(Polygon(outer, holes))
    return polys


def load_aoi(path: str | Path, name: str | None = None) -> AOI:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".kmz":
        with zipfile.ZipFile(path) as z:
            kml_name = next(n for n in z.namelist() if n.lower().endswith(".kml"))
            polys = _polygons_from_kml(z.read(kml_name).decode("utf-8", errors="ignore"))
    elif suffix == ".kml":
        polys = _polygons_from_kml(path.read_text(encoding="utf-8", errors="ignore"))
    else:
        gdf = gpd.read_file(path)
        gdf = gdf.to_crs(4326) if gdf.crs else gdf.set_crs(4326)
        polys = [g for g in gdf.geometry if g is not None and g.geom_type in ("Polygon", "MultiPolygon")]

    if not polys:
        raise ValueError(f"No polygon found in {path}")

    geom = make_valid(unary_union(polys))
    # Drop Z values so downstream geometry maths stays 2D.
    geom = shp_transform(lambda x, y, z=None: (x, y), geom)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        geom = unary_union([g for g in getattr(geom, "geoms", [geom]) if g.geom_type in ("Polygon", "MultiPolygon")])
    return AOI(name=name or path.stem, geometry=geom, source=str(path))
