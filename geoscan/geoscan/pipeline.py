"""End-to-end Phase-1 pipeline: acquire -> EMIT L2A -> EMIT L2B -> ASTER GED -> structure -> maps.

Steps communicate only through files in the workspace, so any step can be re-run on
its own (e.g. from a dashboard) and every output is listed in ``outputs/<aoi>/manifest.json``.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import __version__
from .aoi import AOI, load_aoi
from .config import Settings
from .grid import grid_for_aoi
from .raster_io import aoi_mask, read_raster, write_raster
from .workspace import Workspace

log = logging.getLogger(__name__)

STEPS = ("acquire", "emit_l2a", "emit_l2b", "aster", "structure", "alteration", "targets", "maps")


class Pipeline:
    def __init__(self, aoi_path: str | Path, name: str, root: str | Path, settings: Settings,
                 progress: Callable[[str, str], None] | None = None, output_name: str | None = None):
        self.settings = settings
        self.ws = Workspace(Path(root), name, output_name)
        self.aoi: AOI = load_aoi(aoi_path, name)
        self.progress = progress or (lambda step, msg: log.info("[%s] %s", step, msg))

        g = settings.grid
        self.grid30 = grid_for_aoi(self.aoi, g.base_res_m, g.align_m)
        self.grid_emit = self.grid30.coarsen(g.emit_factor)
        self.grid_aster = self.grid30.coarsen(g.aster_factor)
        self.grid_dem = self.grid30.pad(g.structure_pad_m)
        self.aoi_utm = self.aoi.to_crs(self.grid30.crs)
        self._outputs: dict[str, dict] = {}
        self._step_info: dict[str, dict] = {}

    # ------------------------------------------------------------ helpers
    def _record(self, path: Path, description: str, **extra) -> Path:
        rel = path.relative_to(self.ws.outputs).as_posix()
        self._outputs[rel] = {"description": description, **extra}
        return path

    def _raster(self, rel: str, data, grid, names, description: str, **kw) -> Path:
        return self._record(write_raster(self.ws.out(rel), data, grid, names, **kw), description,
                            bands=list(names), grid=grid.to_dict())

    def _table(self, rel: str, df: pd.DataFrame, description: str) -> Path:
        p = self.ws.out(rel)
        df.to_csv(p, index=False)
        return self._record(p, description, rows=len(df))

    # ------------------------------------------------------------ run
    def run(self, steps: tuple[str, ...] = STEPS, clean: bool | None = None) -> dict:
        unknown = set(steps) - set(STEPS)
        if unknown:
            raise ValueError(f"Unknown steps: {sorted(unknown)}")
        if clean is None:
            clean = set(steps) == set(STEPS)  # full runs start from an empty outputs folder
        if clean:
            self.ws.clean_outputs()
        self._load_manifest()
        self._save_aoi()
        for step in STEPS:
            if step in steps:
                self.progress(step, "started")
                t0 = datetime.now(timezone.utc)
                info = getattr(self, f"step_{step}")() or {}
                info["seconds"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 1)
                self._step_info[step] = info
                self.progress(step, f"done in {info['seconds']} s")
                self._write_manifest()
        return self._write_manifest()

    def _save_aoi(self):
        self.ws.aoi_dir.mkdir(parents=True, exist_ok=True)
        src = Path(self.aoi.source)
        if src.parent.resolve() != self.ws.aoi_dir.resolve():
            shutil.copy2(src, self.ws.aoi_dir / src.name)
        self.aoi.to_geojson(self.ws.aoi_dir / "aoi.geojson")
        self.aoi.to_geojson(self._record(self.ws.out("vectors", "aoi.geojson"), "Area of interest (EPSG:4326)"))

    def _load_manifest(self):
        p = self.ws.outputs / "manifest.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            self._outputs = m.get("outputs", {})
            self._step_info = m.get("steps", {})

    def _write_manifest(self) -> dict:
        import geoscan
        manifest = {
            "run": {"output_name": self.ws.output_name, "label": self.settings.run_label or "main product"},
            "aoi": {"name": self.aoi.name, "source": self.aoi.source, "area_km2": round(self.aoi.area_km2(), 3),
                    "bounds_wgs84": list(self.aoi.bounds), "utm_crs": self.grid30.crs},
            "grids": {"terrain_30m": self.grid30.to_dict(), "emit_60m": self.grid_emit.to_dict(),
                      "aster_90m": self.grid_aster.to_dict(), "dem_padded_30m": self.grid_dem.to_dict()},
            "settings": self.settings.to_dict(),
            "steps": self._step_info,
            "outputs": dict(sorted(self._outputs.items())),
            "software": {"geoscan": __version__, "python": platform.python_version(),
                         "numpy": np.__version__, "package_path": str(Path(geoscan.__file__).parent)},
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.ws.outputs.mkdir(parents=True, exist_ok=True)
        (self.ws.outputs / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return manifest

    # ------------------------------------------------------------ steps
    def step_acquire(self) -> dict:
        from .acquire import gee
        a = self.settings.acquire
        gee.initialize(a.ee_project)
        scenes = gee.list_emit_scenes(self.aoi, a.start_date, a.end_date)
        df = pd.DataFrame(scenes)
        df["selected"] = df["aoi_coverage"] >= a.min_aoi_coverage if len(df) else []
        raw = self.ws.raw("emit_l2a")
        raw.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw / "scenes_available.csv", index=False)
        self.progress("acquire", f"{len(df)} EMIT L2A scenes found, {int(df['selected'].sum()) if len(df) else 0} selected")

        for i, s in enumerate(df[df["selected"]].itertuples() if len(df) else []):
            self.progress("acquire", f"EMIT {s.scene_id} ({s.datetime[:10]}, {100 * s.aoi_coverage:.0f}% of AOI)")
            gee.download_emit_scene(s.scene_id, self.grid_emit, raw, a.max_request_mb, a.retries)
        src = self.settings.structure.dem_source
        self.progress("acquire", f"{gee.dem_label(src)} DEM")
        gee.download_dem(src, self.grid_dem, gee.dem_path(self.ws.raw(""), src), a.max_request_mb, a.retries)
        self.progress("acquire", "ASTER GED")
        gee.download_aster_ged(self.grid_aster, self.ws.raw("aster_ged") / "ged_90m.tif", a.max_request_mb, a.retries)
        return {"scenes_found": len(df), "scenes_selected": int(df["selected"].sum()) if len(df) else 0}

    def _selected_scene_dirs(self) -> list[Path]:
        raw = self.ws.raw("emit_l2a")
        table = raw / "scenes_available.csv"
        if not table.exists():
            raise FileNotFoundError("Run the 'acquire' step first")
        df = pd.read_csv(table)
        dirs = [raw / sid for sid in df.loc[df["selected"], "scene_id"]]
        missing = [d.name for d in dirs if not (d / "meta.json").exists()]
        if missing:
            raise FileNotFoundError(f"EMIT scenes not downloaded: {missing}")
        return dirs

    def step_emit_l2a(self) -> dict:
        from .spectral.composite import COMPOSITE_BANDS, ancillary_composite, composite
        from .spectral.features import FEATURES
        from .spectral.scene import QA_LABELS, process_scene

        cfg = self.settings.spectral
        inside = aoi_mask(self.grid_emit, self.aoi_utm)
        avail = pd.read_csv(self.ws.raw("emit_l2a") / "scenes_available.csv")
        results = []
        for d in self._selected_scene_dirs():
            self.progress("emit_l2a", f"processing {d.name}")
            out = self.ws.processed("emit_l2a") / f"{d.name}_features.tif"
            results.append(process_scene(d, inside, cfg, out))

        scene_df = pd.DataFrame([r.stats for r in results]).merge(
            avail[["scene_id", "aoi_coverage", "solar_azimuth"]], on="scene_id", how="left")
        self._table("tables/emit_l2a_scenes.csv", scene_df, "EMIT L2A scenes used, coverage and pixel quality")

        for r in results:
            self._raster(f"rasters/emit_l2a/scenes/{r.scene_id}_qa.tif", r.layers["qa"], self.grid_emit, ["qa_class"],
                         "Per-scene pixel quality class: " + "; ".join(f"{k}={v}" for k, v in QA_LABELS.items()),
                         nodata=255, dtype="uint8")

        comp = composite(results, cfg)
        summary = []
        for feat in FEATURES:
            layers = comp[feat.key]
            self._raster(f"rasters/emit_l2a/{feat.key}.tif", np.stack([layers[b] for b in COMPOSITE_BANDS]),
                         self.grid_emit, COMPOSITE_BANDS,
                         f"{feat.label}: multi-date band depth with uncertainty, detections and position (nm). "
                         f"Typical causes: {feat.minerals}",
                         tags={"feature": feat.key, "left_nm": feat.left_nm, "right_nm": feat.right_nm,
                               "window_nm": list(feat.window_nm)})
            det = layers["detected"] == 1
            valid = np.isfinite(layers["n_valid"])
            summary.append({
                "feature": feat.key, "label": feat.label, "typical_minerals": feat.minerals,
                "valid_pixels": int(valid.sum()), "detected_pixels": int(det.sum()),
                "detected_pct_of_valid": round(100 * det.sum() / max(valid.sum(), 1), 2),
                "detected_area_km2": round(det.sum() * self.grid_emit.res ** 2 / 1e6, 3),
                "depth_median_detected": _q(layers["depth"][det], 50),
                "depth_p90_detected": _q(layers["depth"][det], 90),
                "snr_median_detected": _q(layers["snr"][det], 50),
                "position_median_nm": _q(layers["position"][det], 50),
                "position_p10_nm": _q(layers["position"][det], 10),
                "position_p90_nm": _q(layers["position"][det], 90),
            })
        self._table("tables/emit_l2a_feature_summary.csv", pd.DataFrame(summary),
                    "Per-feature detection statistics inside the AOI")

        anc = ancillary_composite(results)
        anc_names = list(anc)
        self._raster("rasters/emit_l2a/quality_vegetation.tif", np.stack([anc[n] for n in anc_names]),
                     self.grid_emit, anc_names, "Number of valid scenes, NDVI median/min, cellulose absorption index")
        return {"scenes": len(results), "features": [f.key for f in FEATURES]}

    def step_emit_l2b(self) -> dict:
        from .spectral.l2b import CLASS_KEYS, CLASS_LABELS, process_l2b
        folder = self.ws.raw("emit_l2b")
        if not folder.exists() or not any(folder.glob("*L2B_MIN_*.nc")):
            self.progress("emit_l2b", "no L2B files in raw/emit_l2b - skipped")
            return {"skipped": True}
        inside = aoi_mask(self.grid_emit, self.aoi_utm)
        layers, stats, lib_rows, n = process_l2b(folder, self.grid_emit, inside, self.settings.l2b)

        present = [k for k in CLASS_KEYS if np.nansum(layers[k]["n_dates"]) > 0]
        for var, desc in (("n_dates", "Number of scenes in which EMIT L2B identifies each class"),
                          ("max_depth", "Maximum EMIT L2B band depth per class"),
                          ("mean_fit", "Mean EMIT L2B Tetracorder fit per class"),
                          ("weight", "Sum over dates of EMIT L2B fit / band-depth uncertainty per class"),
                          ("weighted_depth", "EMIT L2B band depth weighted by fit / uncertainty per class")):
            self._raster(f"rasters/emit_l2b/l2b_{var}.tif", np.stack([layers[k][var] for k in present]),
                         self.grid_emit, present, desc + " (bands = classes)")

        stack = np.stack([np.nan_to_num(layers[k]["n_dates"]) for k in present])
        dominant = np.where(stack.max(axis=0) > 0, np.argmax(stack, axis=0), -1).astype("float32")
        dominant = np.where(inside, dominant, np.nan)
        dom_codes = {i: k for i, k in enumerate(present)}
        self._raster("rasters/emit_l2b/l2b_dominant_class.tif", dominant, self.grid_emit, ["class_code"],
                     "Most frequently identified L2B class per pixel (-1 = none). Codes: " +
                     "; ".join(f"{i}={k}" for i, k in dom_codes.items()))

        rows = []
        for k in present:
            nd = layers[k]["n_dates"]
            rows.append({"class": k, "label": CLASS_LABELS[k], "pixels_any_date": int((nd >= 1).sum()),
                         "pixels_2plus_dates": int((nd >= 2).sum()),
                         "area_km2_any_date": round((nd >= 1).sum() * self.grid_emit.res ** 2 / 1e6, 3),
                         "median_max_depth": _q(layers[k]["max_depth"][nd >= 1], 50),
                         "median_fit": _q(layers[k]["mean_fit"][nd >= 1], 50)})
        self._table("tables/emit_l2b_class_summary.csv", pd.DataFrame(rows).sort_values("pixels_any_date", ascending=False),
                    "EMIT L2B mineral classes inside the AOI")
        self._table("tables/emit_l2b_library_classes.csv", pd.DataFrame(lib_rows),
                    "Every Tetracorder library entry found and the class it was assigned to (audit table)")
        self._table("tables/emit_l2b_granules.csv", pd.DataFrame(stats).fillna(0), "L2B granules processed")

        # Cross-check with L2A features where both exist
        xrows = []
        for feat_path in sorted((self.ws.outputs / "rasters" / "emit_l2a").glob("*.tif")):
            if feat_path.stem == "quality_vegetation":
                continue
            data, _, names, _ = read_raster(feat_path)
            det = data[names.index("detected")] == 1
            if not det.any():
                continue
            row = {"l2a_feature": feat_path.stem, "l2a_detected_pixels": int(det.sum())}
            for k in present:
                row[f"pct_with_l2b_{k}"] = round(100 * float((layers[k]["n_dates"][det] >= 1).mean()), 2)
            xrows.append(row)
        if xrows:
            self._table("tables/l2a_vs_l2b_crosscheck.csv", pd.DataFrame(xrows),
                        "For pixels where an L2A feature is detected: % also identified by L2B, per class")
        return {"granules": n, "classes_present": present}

    def step_aster(self) -> dict:
        from .thermal.aster_ged import compute_indices
        layers, grid = compute_indices(self.ws.raw("aster_ged") / "ged_90m.tif", self.settings.spectral.mc_draws,
                                       self.settings.spectral.seed)
        inside = aoi_mask(grid, self.aoi_utm)
        names = list(layers)
        self._raster("rasters/aster_ged/tir_indices.tif", np.stack([layers[n] for n in names]), grid, names,
                     "ASTER GED Ninomiya indices (quartz, carbonate, mafic) with Monte-Carlo spread from GED sdev")
        rows = [{"index": n, "median": _q(layers[n][inside], 50), "p10": _q(layers[n][inside], 10),
                 "p90": _q(layers[n][inside], 90)} for n in ("quartz_index", "silica_index", "carbonate_index", "mafic_index")]
        self._table("tables/aster_ged_index_summary.csv", pd.DataFrame(rows), "ASTER GED index statistics in the AOI")
        return {}

    def step_structure(self) -> dict:
        from .acquire.gee import dem_label, dem_path
        from .structure.lineaments import extract_lineaments, lineament_rasters, rose_table
        src = self.settings.structure.dem_source
        dem, dem_grid, _, _ = read_raster(dem_path(self.ws.raw(""), src))
        dem = dem[0]
        gdf, terr, stats = extract_lineaments(dem, dem_grid, self.aoi_utm, self.settings.structure)
        sc = self.settings.structure
        structural = gdf[gdf["morphology"] != "scarp"] if (sc.exclude_scarps and len(gdf)) else gdf
        rs = lineament_rasters(structural, dem_grid, sc.density_radius_m)
        stats["used_for_proximity"] = len(structural)
        stats["morphology"] = gdf["morphology"].value_counts().to_dict() if len(gdf) else {}
        win = self.grid30.window_in(dem_grid)

        self._raster("rasters/terrain/dem.tif", dem[win], self.grid30, ["elevation_m"], f"{dem_label(src)} elevation (m)")
        self._raster("rasters/terrain/hillshade.tif", terr["hillshade"][win], self.grid30, ["hillshade"],
                     "Hillshade (azimuth 315, elevation 45)")
        self._raster("rasters/terrain/slope.tif", terr["slope_deg"][win], self.grid30, ["slope_deg"], "Slope (degrees)")
        self._raster("rasters/structure/lineament_distance.tif", rs["distance_m"][win], self.grid30, ["distance_m"],
                     "Distance to nearest lineament (m), computed on padded grid")
        self._raster("rasters/structure/lineament_density.tif", rs["density_km_per_km2"][win], self.grid30,
                     ["density_km_per_km2"],
                     f"Lineament density (km/km^2) within {self.settings.structure.density_radius_m:.0f} m")

        gpkg = self.ws.out("vectors", "lineaments.gpkg")
        gdf.to_file(gpkg, driver="GPKG")
        self._record(gpkg, f"Lineaments in {self.grid30.crs}", features=len(gdf))
        gj = self.ws.out("vectors", "lineaments.geojson")
        gdf.to_crs(4326).to_file(gj, driver="GeoJSON")
        self._record(gj, "Lineaments (EPSG:4326)", features=len(gdf))
        self._table("tables/lineaments.csv", gdf.drop(columns="geometry"), "Lineament attributes")
        self._table("tables/lineament_rose.csv", rose_table(gdf), "Length-weighted strike histogram (10 deg bins)")
        stats["dem_source"] = src
        return stats

    def _to_base_grid(self, arr: np.ndarray, src_grid) -> np.ndarray:
        """Bilinear resampling of a smooth layer onto the 30 m analysis grid (shared origin)."""
        from rasterio.warp import Resampling, reproject
        dst = np.full(self.grid30.shape, np.nan, dtype="float32")
        reproject(arr.astype("float32"), dst, src_transform=src_grid.transform, src_crs=src_grid.crs,
                  dst_transform=self.grid30.transform, dst_crs=self.grid30.crs, resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        return dst

    def _zones(self, arr: np.ndarray, grid, tiers: tuple, names: tuple, rel: str, label: str) -> None:
        import geopandas as gpd
        from rasterio.features import shapes
        from shapely.geometry import shape
        cls = np.zeros(grid.shape, dtype="int16")
        for i in range(len(tiers) - 1):
            cls[(arr >= tiers[i]) & (arr < tiers[i + 1])] = i + 1
        recs, geoms = [], []
        for geom, val in shapes(cls, mask=cls > 0, transform=grid.transform):
            v = int(val)
            geoms.append(shape(geom))
            recs.append({"tier_id": v, "class": names[v - 1], "min_value": tiers[v - 1],
                         "max_value": min(tiers[v], 1.0), "layer": label})
        gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs=grid.crs)
        if len(gdf):
            gdf["area_km2"] = (gdf.area / 1e6).round(4)
        path = self.ws.out(rel)
        gdf.to_crs(4326).to_file(path, driver="GeoJSON")
        self._record(path, f"{label} zones by tier (EPSG:4326)", features=len(gdf))

    def _lithology(self):
        """Lithology polygons (grid CRS) for background removal, or None if none is available."""
        import geopandas as gpd
        cfg = self.settings.alteration
        path = Path(cfg.lithology_file) if cfg.lithology_file else \
            next(iter(sorted((self.ws.data / "ground_truth").glob("lithology*.geojson"))), None)
        if path is None or not Path(path).exists():
            self.progress("alteration", "lithology_background requested but no lithology file found - skipped")
            return None
        lith = gpd.read_file(path).to_crs(self.grid30.crs)
        lith = lith[lith.geometry.intersects(self.aoi_utm)]
        self.progress("alteration", f"lithology background from {Path(path).name} ({len(lith)} polygons)")
        return lith

    def step_alteration(self) -> dict:
        """Muscovite, chlorite, gossan, quartz layers (methodology sections 3-4)."""
        from .prospectivity.alteration import (LABELS, MINERALS, ZONE_TIERS, data_support, emit_evidence,
                                               normalise, potential_field, quartz_layer)
        from .thermal.aster_ged import quartz_silicification
        cfg = self.settings.alteration
        out = self.ws.outputs

        feats = {}
        for key in ("ferric_900", "aloh_2200", "feoh_2250", "mgoh_2330"):
            data, grid_e, names, _ = read_raster(out / f"rasters/emit_l2a/{key}.tif")
            feats[key] = {n: data[i] for i, n in enumerate(names)}
        l2b = None
        nd_path = out / "rasters/emit_l2b/l2b_n_dates.tif"
        if nd_path.exists():
            nd, _, classes, _ = read_raster(nd_path)
            l2b = {k: nd[i] for i, k in enumerate(classes)}

        inside_e = aoi_mask(self.grid_emit, self.aoi_utm)
        support = np.where(inside_e, data_support(feats["aloh_2200"]["n_valid"], self.grid_emit.res, cfg), np.nan)
        evidence = emit_evidence(feats, l2b, cfg)
        if cfg.l2b_evidence != "none" and (out / "rasters/emit_l2b/l2b_weight.tif").exists():
            from .prospectivity.alteration import add_l2b_evidence
            wt, _, wcls, _ = read_raster(out / "rasters/emit_l2b/l2b_weight.tif")
            wd, _, _, _ = read_raster(out / "rasters/emit_l2b/l2b_weighted_depth.tif")
            l2a_valid = np.nan_to_num(feats["aloh_2200"]["n_valid"]) > 0
            evidence = add_l2b_evidence(evidence, {k: wt[i] for i, k in enumerate(wcls)},
                                        {k: wd[i] for i, k in enumerate(wcls)}, l2a_valid, cfg)

        lith = self._lithology() if cfg.lithology_background else None
        if lith is not None:
            from .prospectivity.alteration import lithology_background, unit_raster
            units_e, unit_names = unit_raster(lith, cfg.lithology_field, self.grid_emit)
            min_px_e = int(cfg.min_unit_area_km2 * 1e6 / self.grid_emit.res ** 2)
            bg_rows = []

        a30, rows = {}, []
        for m in ("muscovite", "chlorite", "gossan"):
            ev = evidence[m]
            field = potential_field(ev["depth"], ev["weight"], self.grid_emit.res, cfg)
            if lith is not None:
                valid = inside_e & (np.nan_to_num(support) >= cfg.min_support)
                field, bgs = lithology_background(field, units_e, valid, min_px_e)
                bg_rows += [{"layer": m, "unit": unit_names.get(u, "unmapped"), "background": round(b, 5)}
                            for u, b in bgs.items()]
            a, ref = normalise(field, inside_e, cfg.reference_percentile)
            self._raster(f"rasters/alteration/{m}.tif",
                         np.stack([a, np.where(inside_e, field, np.nan), ev["detected"].astype("float32"), support]),
                         self.grid_emit, ["intensity", "potential_field", "detected", "data_support"],
                         f"{LABELS[m]}: relative absorption intensity 0-1 (100 % = P{cfg.reference_percentile:g} "
                         f"of field = {ref:.4f})", tags={"reference": ref})
            a30[m] = self._to_base_grid(a, self.grid_emit)
            rows.append({"mineral": m, "label": LABELS[m], "source": "EMIT L2A (+ L2B for carbonate exclusion)",
                         "detected_pixels_60m": int((ev["detected"] & inside_e).sum()), "reference_value": round(ref, 5)})

        ged, grid_a, gnames, _ = read_raster(out / "rasters/aster_ged/tir_indices.tif")
        inside_a = aoi_mask(grid_a, self.aoi_utm)
        qsi = quartz_silicification(ged[gnames.index("quartz_index")], ged[gnames.index("silica_index")],
                                    inside_a, *cfg.quartz_weights)
        if lith is not None:
            units_a, unit_names_a = unit_raster(lith, cfg.lithology_field, grid_a)
            qsi_res, bgs = lithology_background(qsi, units_a, inside_a,
                                                int(cfg.min_unit_area_km2 * 1e6 / grid_a.res ** 2))
            a_q, _ = normalise(qsi_res, inside_a, cfg.reference_percentile)
            bg_rows += [{"layer": "quartz", "unit": unit_names_a.get(u, "unmapped"), "background": round(b, 5)}
                        for u, b in bgs.items()]
            self._table("tables/lithology_backgrounds.csv", pd.DataFrame(bg_rows),
                        "Per-unit background (median) removed from each alteration layer")
        else:
            a_q = quartz_layer(qsi, inside_a, cfg)
        self._raster("rasters/alteration/quartz.tif", np.stack([a_q, qsi]), grid_a, ["intensity", "qsi"],
                     f"{LABELS['quartz']}: ASTER GED QSI = {cfg.quartz_weights[0]:g} QI + {cfg.quartz_weights[1]:g} SI; "
                     f"intensity = (QSI - P{cfg.quartz_background_percentile:g})/(P{cfg.reference_percentile:g} - "
                     f"P{cfg.quartz_background_percentile:g})")
        a30["quartz"] = self._to_base_grid(a_q, grid_a)
        rows.append({"mineral": "quartz", "label": LABELS["quartz"], "source": "ASTER GED (QI, SI)",
                     "detected_pixels_60m": None, "reference_value": None})

        sup30 = self._to_base_grid(support, self.grid_emit)
        inside30 = aoi_mask(self.grid30, self.aoi_utm)
        stack = [np.where(inside30, a30[m], np.nan) for m in MINERALS] + [np.where(inside30, sup30, np.nan)]
        self._raster("rasters/alteration/alteration_stack_30m.tif", np.stack(stack), self.grid30,
                     list(MINERALS) + ["data_support"], "Four alteration intensities (0-1) and EMIT data support, 30 m")

        for m in MINERALS:
            tiers, names = ZONE_TIERS[m]
            self._zones(np.nan_to_num(a30[m]) * inside30, self.grid30, tiers, names,
                        f"vectors/alteration_{m}_zones.geojson", LABELS[m])
            v = a30[m][inside30]
            r = next(x for x in rows if x["mineral"] == m)
            for i, (t, name) in enumerate(zip(tiers[:-1], names), start=1):
                r[f"zone{i}_name"] = name
                r[f"zone{i}_min_pct"] = int(round(100 * t))
                r[f"zone{i}_area_km2"] = round(float((v >= t).sum()) * self.grid30.res ** 2 / 1e6, 3)
        self._table("tables/alteration_summary.csv", pd.DataFrame(rows), "Alteration layers: sources and areas by tier")
        return {"minerals": list(MINERALS)}

    def step_targets(self) -> dict:
        """CPI, ranked prospective zones and suggested field-check locations (methodology sections 5-6)."""
        import geopandas as gpd
        from shapely.geometry import Point
        from .prospectivity.targets import cpi, prospective_zones, select_locations
        cfg = self.settings.targets
        out = self.ws.outputs

        stack, grid, names, _ = read_raster(out / "rasters/alteration/alteration_stack_30m.tif")
        a = {n: np.nan_to_num(stack[i]) for i, n in enumerate(names)}
        dist, _, _, _ = read_raster(out / "rasters/structure/lineament_distance.tif")
        dist = dist[0]
        s_prox = np.exp(-np.nan_to_num(dist, nan=1e9) / cfg.structure_lambda_m).astype("float32")
        inside = aoi_mask(grid, self.aoi_utm)
        c, h = cpi(a, s_prox, inside, cfg)
        self._raster("rasters/targets/cpi.tif", np.stack([c, h, np.where(inside, s_prox, np.nan)]), grid,
                     ["cpi", "hydrothermal_index", "structural_proximity"],
                     f"Copper Prospectivity Index: CPI = {cfg.w_hydrothermal:g} H + {cfg.w_structure:g} S_prox, "
                     f"H = {cfg.w_muscovite:g} musc + {cfg.w_quartz:g} qtz + {cfg.w_gossan:g} goss + "
                     f"{cfg.w_chlorite:g} chl, S_prox = exp(-d/{cfg.structure_lambda_m:g} m)")

        # ---- prospective zones: the main result (a region, not a point)
        lines = None
        lp = out / "vectors/lineaments.gpkg"
        if lp.exists():
            lines = gpd.read_file(lp).to_crs(grid.crs)
            if self.settings.structure.exclude_scarps and len(lines):
                lines = lines[lines["morphology"] != "scarp"]
        zones, zone_rank = prospective_zones(c, grid, a, s_prox, cfg, lines, a["data_support"],
                                             self.settings.alteration.min_support)

        # ---- suggested field-check locations: where to start looking inside the zones
        eligible = inside & ((a["data_support"] >= self.settings.alteration.min_support)
                             if cfg.require_data_support else True)
        attrs = {"Muscovite_Pct": 100 * a["muscovite"], "Quartz_Pct": 100 * a["quartz"],
                 "Gossan_Pct": 100 * a["gossan"], "Chlorite_Pct": 100 * a["chlorite"],
                 "Hydrothermal_Index": h, "Structural_Proximity": s_prox, "Distance_to_Lineament_m": dist,
                 "EMIT_Data_Support": a["data_support"]}
        df = select_locations(c, grid, attrs, eligible, cfg, zone_rank)
        for col in ("Muscovite_Pct", "Quartz_Pct", "Gossan_Pct", "Chlorite_Pct"):
            if col in df:
                df[col] = df[col].round(1)
        if "Distance_to_Lineament_m" in df:
            df["Distance_to_Lineament_m"] = df["Distance_to_Lineament_m"].round(0)
        zones["Field_Check_Locations"] = [int((df["Zone"] == z).sum()) if len(df) else 0 for z in zones["Zone_ID"]] \
            if len(zones) else []

        for stale in ("tables/top50_targets.csv", "vectors/top50_targets.geojson"):    # names used before zones
            (out / stale).unlink(missing_ok=True)
            self._outputs.pop(stale, None)
        self._table("tables/prospective_zones.csv", pd.DataFrame(zones.drop(columns="geometry")),
                    f"Prospective zones: connected areas with CPI >= {cfg.tier2:g} (Tier 1 / 2) and isolated anomalies "
                    f"with CPI >= {cfg.tier3:g} (Tier 3), ranked by tier and peak CPI")
        zp = self.ws.out("vectors", "prospective_zones.geojson")
        zones.to_crs(4326).to_file(zp, driver="GeoJSON")
        self._record(zp, "Prospective zones, ranked (EPSG:4326)", features=len(zones))
        self._table("tables/field_check_locations.csv", df,
                    f"Suggested field-check locations (highest CPI cells, >= {cfg.min_spacing_m:g} m apart). "
                    "Starting points for field work inside the zones, not drill collars.")
        fp = self.ws.out("vectors", "field_check_locations.geojson")
        if len(df):
            gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["Longitude"], df["Latitude"])], crs=4326)
            gdf.to_file(fp, driver="GeoJSON")
            self._record(fp, "Suggested field-check locations (EPSG:4326)", features=len(gdf))
        else:
            fp.unlink(missing_ok=True)
        self._zones(np.nan_to_num(c), grid, (cfg.tier3, cfg.tier2, cfg.tier1, 1.001),
                    ("Tier 3 General Anomaly", "Tier 2 Medium Priority", "Tier 1 High Priority"),
                    "vectors/cpi_target_zones.geojson", "CPI tier bands")
        tiers = zones["Priority_Tier"].value_counts().to_dict() if len(zones) else {}
        return {"zones": len(zones), "zone_tiers": tiers, "zone_area_km2": round(float(zones["Area_km2"].sum()), 3)
                if len(zones) else 0.0, "field_check_locations": len(df), "cpi_max": round(float(np.nanmax(c)), 4)}

    def step_maps(self) -> dict:
        from .report import render
        n = render.render_all(self)
        return {"maps": n}


def _q(values: np.ndarray, q: float):
    v = values[np.isfinite(values)]
    return round(float(np.percentile(v, q)), 5) if v.size else None
