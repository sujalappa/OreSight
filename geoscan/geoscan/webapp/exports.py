"""Downloadable items for a run and the zip builder.

Same content rules as the reviewer package: main maps only (no diagnostics), rasters without the
internal data-support / NDVI bands, tables and vectors without the EMIT_Data_Support column.
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ..raster_io import read_raster, write_raster
from . import catalog, geochem, geophysics, layers

MAPS = {"cpi_target_map.png": "CPI map: prospective zones and suggested field-check locations",
        "alteration_muscovite_map.png": "Muscovite / sericite (phyllic) contour map",
        "alteration_chlorite_map.png": "Chlorite / epidote (propylitic) contour map",
        "alteration_gossan_map.png": "Gossan / iron oxide contour map",
        "alteration_quartz_map.png": "Quartz / silicification contour map",
        "structure_lineaments_map.png": "Structural lineament map",
        "structure_rose_diagram.png": "Lineament rose diagram"}
RASTERS = {"targets/cpi.tif": "CPI, hydrothermal index, structural proximity",
           "alteration/muscovite.tif": "Muscovite intensity and potential field (60 m)",
           "alteration/chlorite.tif": "Chlorite intensity and potential field (60 m)",
           "alteration/gossan.tif": "Gossan intensity and potential field (60 m)",
           "alteration/quartz.tif": "Quartz intensity and QSI (90 m)",
           "alteration/alteration_stack_30m.tif": "All four alteration intensities (30 m)",
           "structure/lineament_distance.tif": "Distance to nearest lineament (m)",
           "structure/lineament_density.tif": "Lineament density (km/km²)",
           "terrain/dem.tif": "Elevation (m)", "terrain/hillshade.tif": "Hillshade", "terrain/slope.tif": "Slope (°)",
           "aster_ged/tir_indices.tif": "ASTER GED thermal indices"}
DROP_BANDS = {"data_support", "ged_ndvi", "ged_num_obs"}
TABLES = {"prospective_zones.csv": "Prospective zones, ranked (the main result)",
          "field_check_locations.csv": "Suggested field-check locations (starting points, not drill collars)",
          "top50_targets.csv": "Ranked locations (earlier naming)", "alteration_summary.csv": "Alteration layers and zone areas",
          "lineaments.csv": "Lineament attributes", "lineament_rose.csv": "Lineament strike histogram",
          "aster_ged_index_summary.csv": "ASTER GED index statistics"}
VECTOR_LABELS = {"prospective_zones.geojson": "Prospective zones, ranked (polygons, the main result)",
                 "field_check_locations.geojson": "Suggested field-check locations (points)",
                 "top50_targets.geojson": "Ranked locations (points, earlier naming)",
                 "cpi_target_zones.geojson": "CPI tier bands (polygons)",
                 "lineaments.geojson": "Lineaments (lines)", "aoi.geojson": "Area of interest",
                 "alteration_muscovite_zones.geojson": "Muscovite zones (polygons)",
                 "alteration_chlorite_zones.geojson": "Chlorite zones (polygons)",
                 "alteration_gossan_zones.geojson": "Gossan zones (polygons)",
                 "alteration_quartz_zones.geojson": "Quartz zones (polygons)"}
FORMATS = {"shp": "Shapefile", "geojson": "GeoJSON", "gpkg": "GeoPackage"}


CREDITS = [
    "EMIT L2A reflectance and L2B mineral identification: NASA EMIT mission, NASA Jet Propulsion Laboratory; "
    "distributed by NASA LP DAAC (doi:10.5067/EMIT/EMITL2ARFL.001, doi:10.5067/EMIT/EMITL2BMIN.001).",
    "ASTER Global Emissivity Dataset AG100 v003: NASA Jet Propulsion Laboratory; distributed by NASA LP DAAC "
    "(doi:10.5067/Community/ASTER_GED/AG100.003).",
    "Elevation models: ALOS World 3D 30 m (c) JAXA; SRTM 1 arc-second, NASA / USGS (doi:10.5067/MEaSUREs/SRTM/SRTMGL1.003); "
    "Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS "
    "by the European Union and ESA, all rights reserved (doi:10.5270/ESA-c5d3d65). The one used is named under 'Method options'.",
    "Reference gravity, magnetic and stream sediment data: Geological Survey of India, Ministry of Mines, Government of India (Bhukosh portal).",
    "Validation data: National Geoscience Data Repository (NGDR), Geological Survey of India and Mineral Exploration and "
    "Consultancy Limited (MECL).",
    "Satellite data accessed through Google Earth Engine.",
]


def items(run: catalog.Run) -> list[dict]:
    d = run.path
    out = []
    for f, lab in MAPS.items():
        if (d / "maps" / f).exists():
            out.append({"id": f"map:{f}", "group": "Maps (PNG)", "label": lab})
    for k, spec in layers.RASTER_LAYERS.items():
        if spec["levels"] and (d / spec["file"]).exists():
            out.append({"id": f"contour:{k}", "group": "Contour lines (vector)",
                        "label": f"{spec['label']}: {', '.join(map(str, spec['levels']))} {spec['unit']}"})
    for f, lab in VECTOR_LABELS.items():
        if (d / "vectors" / f).exists():
            out.append({"id": f"vector:{f}", "group": "Vectors", "label": lab})
    for f, lab in RASTERS.items():
        if (d / "rasters" / f).exists():
            out.append({"id": f"raster:{f}", "group": "Rasters (GeoTIFF, UTM)", "label": lab})
    for f, lab in TABLES.items():
        if (d / "tables" / f).exists():
            out.append({"id": f"table:{f}", "group": "Tables (CSV)", "label": lab})
    for ref in geophysics.info(run.block):
        parts = "stations, interpolated surface, contours, map" if ref["has_surface"] else "stations, map"
        out.append({"id": f"ref:{ref['key']}", "group": "Reference geophysics (not used in the CPI)",
                    "label": f"{ref['label']}: {parts} ({ref['stations']} stations)"})
    gc = geochem.info(run.block)
    if gc["available"]:
        out.append({"id": f"ref:{geochem.KEY}", "group": "Reference geochemistry (not used in the CPI)",
                    "label": f"Stream sediment samples: all {len(gc['elements'])} elements, points, "
                             f"Cu / Zn / Pb / Au maps ({gc['samples']} samples)"})
    if run.validation:
        out.append({"id": "validation", "group": "Validation", "label": "Validation against ground truth (summary, maps, tables)"})
    return out


def _short_fields(gdf: gpd.GeoDataFrame):
    """Shapefile field names are limited to 10 characters; make unique short names and a lookup table."""
    names, used = {}, set()
    for c in gdf.columns:
        if c == "geometry":
            continue
        base = c[:10]
        n, k = base, 1
        while n.lower() in used:
            n = f"{base[:10 - len(str(k))]}{k}"
            k += 1
        used.add(n.lower())
        names[c] = n
    return gdf.rename(columns=names), pd.DataFrame({"full_name": list(names), "shapefile_field": list(names.values())})


def _write_vector(z: zipfile.ZipFile, gdf: gpd.GeoDataFrame, stem: str, folder: str, fmts: list[str], utm):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if "geojson" in fmts:
            p = tmp / f"{stem}.geojson"
            gdf.to_crs(4326).to_file(p, driver="GeoJSON")
            z.write(p, f"{folder}/geojson/{p.name}")
        if "gpkg" in fmts:
            p = tmp / f"{stem}.gpkg"
            gdf.to_crs(utm).to_file(p, driver="GPKG", layer=stem)
            z.write(p, f"{folder}/geopackage/{p.name}")
        if "shp" in fmts:
            g, lookup = _short_fields(gdf.to_crs(utm))
            (tmp / "shp").mkdir()
            g.to_file(tmp / "shp" / f"{stem}.shp", driver="ESRI Shapefile")
            for f in (tmp / "shp").iterdir():
                z.write(f, f"{folder}/shapefile/{stem}/{f.name}")
            if (lookup.full_name != lookup.shapefile_field).any():
                z.writestr(f"{folder}/shapefile/{stem}/{stem}_field_names.csv", lookup.to_csv(index=False))


def build_zip(run: catalog.Run, selected: list[str], fmts: list[str]) -> bytes:
    d = run.path
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    utm = m["aoi"]["utm_crs"]
    fmts = [f for f in fmts if f in FORMATS] or ["shp"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        readme = [f"OreSight results - {run.block}", f"Method options: {catalog.option_label(run.options)}",
                  f"Computed: {run.updated}", f"Coordinate system: rasters and Shapefile / GeoPackage in {utm}; "
                  "GeoJSON in WGS84 (EPSG:4326).", "", "Contents:"]
        for it in selected:
            kind, _, name = it.partition(":")
            if kind == "map" and name in MAPS:
                z.write(d / "maps" / name, f"maps/{name}")
            elif kind == "raster" and name in RASTERS:
                data, grid, bands, tags = read_raster(d / "rasters" / name)
                keep = [i for i, b in enumerate(bands) if b not in DROP_BANDS]
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / Path(name).name
                    write_raster(p, data[keep], grid, [bands[i] for i in keep], tags=tags)
                    z.write(p, f"rasters/{name}")
            elif kind == "table" and name in TABLES:
                df = pd.read_csv(d / "tables" / name)
                z.writestr(f"tables/{name}", df.drop(columns=[c for c in df.columns if c in layers.DROP_COLUMNS])
                           .to_csv(index=False))
            elif kind == "vector" and name in VECTOR_LABELS:
                _write_vector(z, layers.vector(d, name.removesuffix(".geojson")), Path(name).stem, "vectors", fmts, utm)
            elif kind == "contour" and name in layers.RASTER_LAYERS:
                g = layers.contours(d, name)
                _write_vector(z, g, f"contours_{name}", "contours", fmts, utm)
            elif kind == "ref" and name == geochem.KEY and geochem.samples(run.block) is not None:
                sm = geochem.samples(run.block)
                folder = "reference_geochemistry/stream_sediment"
                z.writestr(f"{folder}/stream_sediment_samples.csv", sm.drop(columns="geometry").to_csv(index=False))
                _write_vector(z, sm, "stream_sediment_samples", folder, fmts, utm)
                for el in geochem.MAP_ELEMENTS:
                    fig = geochem.figure(run.block, el)
                    if fig is not None:
                        z.write(fig, f"{folder}/stream_sediment_{el}_map.png")
                note = geochem.info(run.block).get("note", "")
                z.writestr(f"{folder}/READ_ME.txt", ("\ufeff" + geochem.LABEL + "\r\n\r\n" + geochem.WHAT + "\r\n\r\n" +
                           geochem.CAVEAT + ("\r\n\r\n" + note if note else "") + "\r\n\r\nSource: " +
                           geochem.SOURCE + "\r\n").encode("utf-8"))
            elif kind == "ref" and name in geophysics.LAYERS and geophysics.stations(run.block, name) is not None:
                st = geophysics.stations(run.block, name)
                folder = f"reference_geophysics/{name}"
                z.write(geophysics.figure(run.block, name), f"{folder}/{name}_map.png")
                z.writestr(f"{folder}/{name}_stations.csv", st.drop(columns="geometry").to_csv(index=False))
                _write_vector(z, st, f"{name}_stations", folder, fmts, utm)
                cont = geophysics.contours(run.block, name)
                if cont is not None:
                    _write_vector(z, cont, f"{name}_contours", folder, fmts, utm)
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / f"{name}_surface.tif"
                    if geophysics.raster_file(run.block, name, p):
                        z.write(p, f"{folder}/{p.name}")
                z.writestr(f"{folder}/READ_ME.txt", ("\ufeff" + geophysics.LAYERS[name]["label"] + "\r\n\r\n" +
                           geophysics.LAYERS[name]["what"] + "\r\n\r\n" + geophysics.CAVEAT + "\r\n\r\nSource: " +
                           geophysics.SOURCE + "\r\n").encode("utf-8"))
            elif kind == "validation" and run.validation:
                for f in run.validation.rglob("*"):
                    if f.is_file():
                        z.write(f, f"validation/{f.relative_to(run.validation).as_posix()}")
            else:
                continue
            readme.append(f"  - {it}")
        readme += ["", "Data sources and credits (keep these with any map or file you pass on):"] + [f"  - {c}" for c in CREDITS]
        z.writestr("README.txt", ("﻿" + "\r\n".join(readme)).encode("utf-8"))   # BOM so Notepad shows UTF-8
    return buf.getvalue()
