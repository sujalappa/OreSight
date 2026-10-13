"""Quality check of every raster and vector in the reviewer package before it is sent.

    python scripts/qa_package.py [package_dir]

Checks, per block:
  rasters   CRS, north-up transform, square pixels, nodata, named bands, no empty band, value ranges,
            DEM matches the independent SRTM tile, all layers aligned on shared grids
  targets   CSV == GeoJSON (ids, coordinates, attributes); CPI in CSV == CPI raster at each point;
            mineral % in CSV == alteration stack; tiers consistent with CPI; spacing >= 250 m;
            all inside the block; ranked by CPI
  vectors   CRS, valid non-empty geometries, inside / intersecting the block, zone polygons agree
            with the raster tier at their interior points, GPKG == GeoJSON lineaments
  hygiene   no internal-only fields (data support, NDVI) anywhere; manifest lists existing files
Exit code 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("data_support", "emit_data_support", "ged_ndvi", "ndvi_median", "n_valid")
RANGES = {  # band -> (min, max) allowed, NaN ignored
    "intensity": (0, 1), "muscovite": (0, 1), "chlorite": (0, 1), "gossan": (0, 1), "quartz": (0, 1),
    "cpi": (0, 1), "hydrothermal_index": (0, 1), "structural_proximity": (0, 1), "qsi": (0, 1),
    "detected": (0, 1), "distance_m": (0, 1e5), "density_km_per_km2": (0, 100), "elevation_m": (-50, 9000),
    "hillshade": (0, 1), "slope_deg": (0, 90), "potential_field": (0, 1),
}
SRTM_LOCAL = {"kehalpur": Path(r"C:\Users\sujal\Downloads\exploration_kelipur\SRTM_data\n21_e078_1arc_v3.tif"),
              "thanewasna": Path(r"C:\Users\sujal\Downloads\exploration_thanewasna\SRTM_data\n19_e079_1arc_v3.tif")}
results: list[tuple[str, str, bool, str]] = []


def check(block, name, ok, detail=""):
    results.append((block, name, bool(ok), detail))


def rasters(block, base):
    grids = {}
    for f in sorted((base / "rasters").rglob("*.tif")):
        rel = f.relative_to(base).as_posix()
        with rasterio.open(f) as s:
            t = s.transform
            check(block, f"{rel}: CRS", s.crs is not None and s.crs.to_epsg() == 32644, str(s.crs))
            check(block, f"{rel}: north-up square pixels", t.b == 0 and t.d == 0 and t.e < 0 and abs(t.a + t.e) < 1e-9,
                  f"{t.a:g} m")
            check(block, f"{rel}: nodata defined", s.nodata is not None, str(s.nodata))
            names = list(s.descriptions)
            check(block, f"{rel}: all bands named", all(names), str(names))
            check(block, f"{rel}: no internal-only bands", not any(n and n.lower() in FORBIDDEN for n in names), str(names))
            for i, n in enumerate(names, start=1):
                a = s.read(i).astype("float64")
                if s.nodata is not None and not np.isnan(s.nodata):
                    a[a == s.nodata] = np.nan
                fin = np.isfinite(a)
                check(block, f"{rel}[{n}]: not empty", fin.mean() > 0.3, f"{100 * fin.mean():.0f}% finite")
                key = n if n in RANGES else ("intensity" if "intensity" in (n or "") else None)
                if key and fin.any():
                    lo, hi = RANGES[key]
                    mn, mx = np.nanmin(a), np.nanmax(a)
                    check(block, f"{rel}[{n}]: range", lo - 1e-6 <= mn and mx <= hi + 1e-6, f"{mn:.3g}..{mx:.3g}")
            grids.setdefault(round(t.a), set()).add((t.c, t.f, s.width, s.height))
    for res, g in grids.items():
        check(block, f"{res} m rasters share one grid", len(g) == 1, str(g))
    origins = {(o[0], o[1]) for g in grids.values() for o in g}
    check(block, "30/60/90 m grids share one origin", len(origins) == 1, str(origins))

    # geolocation: packaged DEM vs independent SRTM tile at 25 random interior points
    if SRTM_LOCAL[block].exists():
        with rasterio.open(base / "rasters/terrain/dem.tif") as d, rasterio.open(SRTM_LOCAL[block]) as ref:
            rng = np.random.default_rng(0)
            b = d.bounds
            xs = rng.uniform(b.left + 200, b.right - 200, 25)
            ys = rng.uniform(b.bottom + 200, b.top - 200, 25)
            lon, lat = Transformer.from_crs(d.crs, ref.crs, always_xy=True).transform(xs, ys)
            v1 = np.array([v[0] for v in d.sample(zip(xs, ys))], float)
            v2 = np.array([v[0] for v in ref.sample(zip(lon, lat))], float)
            err = np.abs(v1 - v2)
            check(block, "DEM geolocation vs independent SRTM tile (25 pts)", np.median(err) <= 2.0,
                  f"median |dz| = {np.median(err):.1f} m, max {err.max():.1f} m")


def sample(path, band, xs, ys):
    with rasterio.open(path) as s:
        i = list(s.descriptions).index(band) + 1
        return np.array([v[i - 1] for v in s.sample(zip(xs, ys))], float)


def targets(block, base, aoi_utm):
    csv = pd.read_csv(base / "tables/top50_targets.csv")
    gj = gpd.read_file(base / "vectors/top50_targets.geojson")
    check(block, "targets: CSV has no internal-only columns", not any(c.lower() in FORBIDDEN for c in csv.columns))
    check(block, "targets: GeoJSON has no internal-only fields", not any(c.lower() in FORBIDDEN for c in gj.columns),
          str([c for c in gj.columns if c.lower() in FORBIDDEN]))
    check(block, "targets: GeoJSON CRS is WGS84", gj.crs is not None and gj.crs.to_epsg() == 4326, str(gj.crs))
    check(block, "targets: same ids in CSV and GeoJSON", list(csv.Target_ID) == list(gj.Target_ID))
    dxy = np.hypot(gj.geometry.x.values - csv.Longitude.values, gj.geometry.y.values - csv.Latitude.values)
    check(block, "targets: GeoJSON points == CSV lon/lat", dxy.max() < 1e-6, f"max diff {dxy.max():.2e} deg")
    tr = Transformer.from_crs(4326, 32644, always_xy=True)
    ex, ny = tr.transform(csv.Longitude.values, csv.Latitude.values)
    d_en = np.hypot(ex - csv.Easting, ny - csv.Northing)
    check(block, "targets: lon/lat == Easting/Northing", d_en.max() < 1.0, f"max {d_en.max():.2f} m")
    cpi = sample(base / "rasters/targets/cpi.tif", "cpi", csv.Easting, csv.Northing)
    check(block, "targets: CSV CPI == CPI raster at point", np.nanmax(np.abs(cpi - csv.CPI)) < 1e-3,
          f"max diff {np.nanmax(np.abs(cpi - csv.CPI)):.2e}")
    for col, band in (("Muscovite_Pct", "muscovite"), ("Quartz_Pct", "quartz"), ("Gossan_Pct", "gossan"),
                      ("Chlorite_Pct", "chlorite")):
        v = 100 * sample(base / "rasters/alteration/alteration_stack_30m.tif", band, csv.Easting, csv.Northing)
        check(block, f"targets: {col} == alteration stack", np.nanmax(np.abs(v - csv[col])) < 0.06,
              f"max diff {np.nanmax(np.abs(v - csv[col])):.3f} %")
    exp_tier = np.where(csv.CPI >= 0.70, "Tier 1", np.where(csv.CPI >= 0.50, "Tier 2",
                        np.where(csv.CPI >= 0.28, "Tier 3", "Below Tier 3")))
    check(block, "targets: tier consistent with CPI",
          all(t.startswith(e) for t, e in zip(csv.Priority_Tier, exp_tier)))
    xy = np.c_[csv.Easting, csv.Northing]
    dd = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    dmin = dd[np.triu_indices(len(xy), 1)].min() if len(xy) > 1 else np.inf
    check(block, "targets: spacing >= 250 m", dmin >= 250 - 1e-6, f"min {dmin:.0f} m")
    inside = gpd.GeoSeries(gpd.points_from_xy(ex, ny), crs=32644).within(aoi_utm.buffer(1))
    check(block, "targets: all inside block", inside.all(), f"{int(inside.sum())}/{len(inside)}")
    check(block, "targets: ranked by CPI", csv.CPI.is_monotonic_decreasing and list(csv.Rank) == list(range(1, len(csv) + 1)))
    check(block, "targets: CPI >= 0.25", (csv.CPI >= 0.25 - 1e-9).all())


def vectors(block, base, aoi_utm):
    for f in sorted((base / "vectors").glob("*")):
        g = gpd.read_file(f)
        rel = f.relative_to(base).as_posix()
        check(block, f"{rel}: has features", len(g) > 0, f"{len(g)} features")
        check(block, f"{rel}: CRS defined", g.crs is not None, str(g.crs))
        check(block, f"{rel}: valid, non-empty geometries", g.geometry.is_valid.all() and not g.geometry.is_empty.any())
        gu = g.to_crs(32644)
        check(block, f"{rel}: within / touching block", gu.intersects(aoi_utm.buffer(5)).all())
        check(block, f"{rel}: no internal-only fields", not any(c.lower() in FORBIDDEN for c in g.columns))
    gp = gpd.read_file(base / "vectors/lineaments.gpkg").to_crs(4326)
    gj = gpd.read_file(base / "vectors/lineaments.geojson")
    same = len(gp) == len(gj) and np.allclose(gp.geometry.length.values, gj.geometry.length.values, rtol=1e-6)
    check(block, "lineaments: GPKG == GeoJSON", same, f"{len(gp)} vs {len(gj)}")
    # zone polygons agree with the raster tier at interior points
    for mineral in ("muscovite", "chlorite", "gossan", "quartz"):
        z = gpd.read_file(base / f"vectors/alteration_{mineral}_zones.geojson").to_crs(32644)
        pts = z.geometry.representative_point()
        v = sample(base / "rasters/alteration/alteration_stack_30m.tif", mineral, pts.x, pts.y)
        ok = (v >= z.min_value - 1e-4) & (v < z.max_value + 1e-4)
        check(block, f"zones {mineral}: polygon tier == raster value", ok.mean() >= 0.98, f"{100 * ok.mean():.0f}% of {len(z)}")
    z = gpd.read_file(base / "vectors/cpi_target_zones.geojson").to_crs(32644)
    pts = z.geometry.representative_point()
    v = sample(base / "rasters/targets/cpi.tif", "cpi", pts.x, pts.y)
    ok = (v >= z.min_value - 1e-4) & (v < z.max_value + 1e-4)
    check(block, "zones CPI: polygon tier == raster value", ok.mean() >= 0.98, f"{100 * ok.mean():.0f}% of {len(z)}")


def manifest(block, base):
    m = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    missing = [k for k in m["outputs"] if not (base / k).exists()]
    check(block, "manifest: every listed file exists", not missing, str(missing[:5]))
    unlisted = [p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p.name != "manifest.json"
                and p.relative_to(base).as_posix() not in m["outputs"]]
    check(block, "manifest: every file is listed", not unlisted, str(unlisted[:5]))


def main():
    pkg = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "deliverables/Phase1_geologist_review_2026-09-24"
    for block, folder in (("kehalpur", "02_Kehalpur"), ("thanewasna", "03_Thanewasna")):
        base = pkg / folder / "A_main_product"
        aoi_utm = gpd.read_file(base / "vectors/aoi.geojson").to_crs(32644).union_all()
        rasters(block, base)
        targets(block, base, aoi_utm)
        vectors(block, base, aoi_utm)
        manifest(block, base)
    df = pd.DataFrame(results, columns=["block", "check", "pass", "detail"])
    out = pkg.parent / "_qa_report.csv"
    df.to_csv(out, index=False)
    fails = df[~df["pass"]]
    print(f"{len(df)} checks, {len(fails)} failed  (report: {out})")
    if len(fails):
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 90)
        print(fails.to_string(index=False))
    sys.exit(1 if len(fails) else 0)


if __name__ == "__main__":
    main()
