"""Render all maps from the files in ``outputs/<aoi>/``.

maps/               main deliverables: 4 alteration maps, CPI target map, lineaments, rose, scene usability
maps/diagnostics/   pixel-level checks: per-feature depth / SNR / persistence, NDVI, valid scenes, L2B classes
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from ..acquire.gee import dem_label
from ..prospectivity.alteration import CONTOUR_LEVELS, LABELS, MINERALS
from ..raster_io import read_raster
from ..spectral.features import FEATURES
from ..spectral.l2b import CLASS_LABELS
from . import maps
from .contour_maps import ContourMap

L2B_COLORS = {
    "white_mica": "#e53935", "kaolinite": "#fb8c00", "smectite": "#fdd835", "alunite": "#8e24aa",
    "pyrophyllite": "#d81b60", "chlorite": "#43a047", "epidote": "#7cb342", "prehnite": "#26a69a",
    "amphibole": "#00897b", "serpentine": "#5e35b1", "talc": "#90a4ae", "carbonate": "#1e88e5",
    "jarosite": "#ffb300", "ferric_oxide": "#6d4c41", "copper_mineral": "#00e5ff", "sulfate": "#bdbdbd",
    "vermiculite": "#a1887f", "biotite": "#3949ab", "pyroxene_olivine": "#1b5e20", "other": "#757575",
}

MINERAL_STYLE = {  # colormap, map subtitle
    "muscovite": ("YlOrRd", "Muscovite / Sericite Absorption Intensity (Phyllic Core) — Al-OH 2200 nm"),
    "chlorite": ("YlGn", "Chlorite / Epidote Absorption Intensity (Propylitic Halo) — Fe-OH 2250 / Mg-OH 2330 nm"),
    "gossan": ("Reds", "Iron Oxide / Hydroxide Absorption Intensity (Gossan Cap) — Fe³⁺ 900 nm"),
    "quartz": ("Greens", "Quartz Silicification Reststrahlen Anomaly — ASTER QSI"),
}


def _band(path, name):
    data, grid, names, _ = read_raster(path)
    return data[names.index(name)], grid


def render_all(p) -> int:
    out = p.ws.outputs
    m, dg = out / "maps", out / "maps" / "diagnostics"
    dg.mkdir(parents=True, exist_ok=True)
    aoi, aoi_ll = p.aoi_utm, p.aoi.geometry
    block = p.aoi.name.replace("_", " ").title()
    if p.settings.run_label:
        block = f"{block} [{p.settings.run_label}]"
    ac, tc = p.settings.alteration, p.settings.targets
    count = 0

    hs_path = out / "rasters/terrain/hillshade.tif"
    bg = _band(hs_path, "hillshade") if hs_path.exists() else None
    lines_path = out / "vectors/lineaments.gpkg"
    lines = gpd.read_file(lines_path) if lines_path.exists() else None
    lines_ll = lines.to_crs(4326) if lines is not None else None
    cmap_ = ContourMap(aoi_ll, bg, lines_ll)
    targets_csv = out / "tables/field_check_locations.csv"
    targets = pd.read_csv(targets_csv) if targets_csv.exists() else None
    zones_csv = out / "tables/prospective_zones.csv"
    zones = pd.read_csv(zones_csv) if zones_csv.exists() else None

    def rec(path, title):
        nonlocal count
        p._record(path, title)
        count += 1

    # ================= 4 alteration maps =================
    for mineral in MINERALS:
        f = out / f"rasters/alteration/{mineral}.tif"
        if not f.exists():
            continue
        data, grid, names, tags = read_raster(f)
        L = {n: data[i] for i, n in enumerate(names)}
        a = 100 * L["intensity"]
        cm, sub = MINERAL_STYLE[mineral]
        support = None
        if "data_support" in L:
            # "no data" only where EMIT had few usable pixels AND nothing was detected there
            support = np.maximum(L["data_support"] / ac.min_support, np.nan_to_num(L["intensity"]) / 0.05)
        path = m / f"alteration_{mineral}_map.png"
        cmap_.surface(path, a, grid, f"{block} Copper Block — Alteration Spectroscopy\n{sub}",
                      f"Relative {LABELS[mineral].split(' /')[0]} Intensity (%)", cmap=cm,
                      levels=CONTOUR_LEVELS[mineral],
                      subtitle=("EMIT L2A multi-date composite · two-tier potential field "
                                f"(σ core {ac.sigma_core_m:.0f} m, halo {ac.sigma_halo_m:.0f} m) · 100 % = P{ac.reference_percentile:g}"
                                if mineral != "quartz" else
                                f"ASTER GED emissivity · QSI = {ac.quartz_weights[0]:g} QI + {ac.quartz_weights[1]:g} SI · "
                                f"(QSI − P{ac.quartz_background_percentile:g})/(P{ac.reference_percentile:g} − "
                                f"P{ac.quartz_background_percentile:g})"),
                      upsample_factor=ac.display_upsample if mineral != "quartz" else 3,
                      support=support, min_support=1.0 if support is not None else None,
                      targets=targets, n_stars=10)
        rec(path, f"Alteration map: {LABELS[mineral]}")

    # ================= CPI target map =================
    cpi_path = out / "rasters/targets/cpi.tif"
    if cpi_path.exists():
        c, grid = _band(cpi_path, "cpi")
        stack_path = out / "rasters/alteration/alteration_stack_30m.tif"
        no_emit = None
        if stack_path.exists():
            # same rule as the mineral maps: few usable EMIT pixels AND no EMIT mineral detected there
            sup, _ = _band(stack_path, "data_support")
            emit_max = np.nanmax(np.stack([np.nan_to_num(_band(stack_path, k)[0]) for k in ("muscovite", "chlorite", "gossan")]), 0)
            no_emit = np.isfinite(sup) & (sup < ac.min_support) & (emit_max < 0.05)
        n = 0 if targets is None else len(targets)
        nz = 0 if zones is None else len(zones)
        tiers = zones["Priority_Tier"].value_counts().to_dict() if nz else {}
        path = m / "cpi_target_map.png"
        cmap_.surface(path, 100 * c, grid,
                      f"{block} Copper Exploration Block — Multi-Sensor Alteration & Targeting Model\n"
                      "Copper Prospectivity Index (CPI): Prospective Zones & Suggested Field-Check Locations",
                      "Integrated Copper Prospectivity Index (CPI, %)", cmap="plasma",
                      levels=(round(100 * tc.tier3), round(100 * tc.tier2), round(100 * tc.tier1)),
                      contour_colors=["#ffffff", "#ffeb3b", "#ff1744"],
                      subtitle=f"CPI = {tc.w_hydrothermal:.2f}·({tc.w_muscovite:.2f} musc + {tc.w_quartz:.2f} qtz + "
                               f"{tc.w_gossan:.2f} goss + {tc.w_chlorite:.2f} chl) + "
                               f"{tc.w_structure:.2f}·exp(−d/{tc.structure_lambda_m:.0f} m) · "
                               f"{nz} zones (" + ", ".join(f"{k.split(':')[0]}: {v}" for k, v in sorted(tiers.items())) +
                               f") · {n} field-check locations ≥ {tc.min_spacing_m:.0f} m apart",
                      upsample_factor=2, targets=targets, n_labels=min(15, n), hatch_mask=no_emit, zones=zones,
                      hatch_label="Alteration not assessed here: CPI from quartz + structure only")
        rec(path, "CPI map with ranked prospective zones and suggested field-check locations")

    # ================= structure =================
    if lines_ll is not None and len(lines_ll):
        path = m / "structure_lineaments_map.png"
        cmap_.lineaments(path, lines_ll, f"{block} — {dem_label(p.settings.structure.dem_source)} Structural Lineaments",
                         subtitle=f"{len(lines_ll)} lineaments ≥ {p.settings.structure.min_length_m:.0f} m; "
                                  f"line width ∝ length")
        rec(path, "Lineament map by strike class")
        rose = pd.read_csv(out / "tables/lineament_rose.csv")
        path = m / "structure_rose_diagram.png"
        maps.rose_diagram(path, rose, f"{block} — lineament strike (length-weighted)")
        rec(path, "Lineament rose diagram")

    scenes_csv = out / "tables/emit_l2a_scenes.csv"
    if scenes_csv.exists():
        path = m / "emit_l2a_scene_usability.png"
        maps.scene_chart(path, pd.read_csv(scenes_csv))
        rec(path, "EMIT L2A scene coverage and usable pixels")

    # ================= diagnostics (pixel level) =================
    def diag(name, arr, grid, title, label, **kw):
        path = dg / name
        maps.layer_map(path, arr, grid, title, label, aoi, **kw)
        rec(path, title.split("\n")[0])

    for feat in FEATURES:
        f = out / f"rasters/emit_l2a/{feat.key}.tif"
        if not f.exists():
            continue
        data, grid, names, _ = read_raster(f)
        L = {n: data[i] for i, n in enumerate(names)}
        det = L["detected"] == 1
        diag(f"emit_{feat.key}_depth_pixels.png", np.where(det, L["depth"], np.nan), grid,
             f"EMIT L2A - {feat.label}: band depth (detected pixels)", "band depth", cmap="inferno", vmin=0,
             background=bg, subtitle=f"{int(det.sum())} detected pixels")
        diag(f"emit_{feat.key}_snr.png", L["snr"], grid, f"EMIT L2A - {feat.label}: signal-to-noise",
             "depth / sigma", cmap="RdBu_r", vmin=-5, vmax=5, background=bg)
        diag(f"emit_{feat.key}_persistence.png", np.where(L["n_valid"] > 0, L["n_detected"], np.nan), grid,
             f"EMIT L2A - {feat.label}: number of dates detected", "scenes", cmap="viridis", vmin=0,
             vmax=max(1, np.nanmax(L["n_valid"]) if np.isfinite(L["n_valid"]).any() else 1), background=bg)
        if np.isfinite(L["position"]).any():
            lo, hi = feat.position_range_nm
            diag(f"emit_{feat.key}_position.png", L["position"], grid,
                 f"EMIT L2A - {feat.label}: wavelength of absorption minimum", "nm", cmap="Spectral_r",
                 vmin=lo, vmax=hi, background=bg)

    q = out / "rasters/emit_l2a/quality_vegetation.tif"
    if q.exists():
        arr, grid = _band(q, "n_valid_scenes")
        diag("emit_valid_scene_count.png", arr, grid, "EMIT L2A - number of valid scenes per pixel",
             "scenes", cmap="viridis", vmin=0)
        arr, grid = _band(q, "ndvi_median")
        diag("emit_ndvi_median.png", arr, grid, "EMIT L2A - median NDVI", "NDVI", cmap="RdYlGn", vmin=-0.1, vmax=0.7)

    dom = out / "rasters/emit_l2b/l2b_dominant_class.tif"
    if dom.exists():
        data, grid, _, _ = read_raster(dom)
        _, _, class_names, _ = read_raster(out / "rasters/emit_l2b/l2b_n_dates.tif")
        path = dg / "emit_l2b_dominant_class.png"
        maps.class_map(path, np.where(data[0] >= 0, data[0], np.nan), grid,
                       {i: CLASS_LABELS[k] for i, k in enumerate(class_names)},
                       {i: L2B_COLORS[k] for i, k in enumerate(class_names)},
                       "EMIT L2B (Tetracorder) - most frequently identified class", aoi, background=bg)
        rec(path, "EMIT L2B dominant mineral class (pixels)")

    for key, label, cm in (("quartz_index", "Quartz index (QI)", "YlGn"), ("silica_index", "Silica index (SI)", "Greens"),
                           ("carbonate_index", "Carbonate index (CI)", "Blues"), ("mafic_index", "Mafic index (MI)", "PuRd")):
        t = out / "rasters/aster_ged/tir_indices.tif"
        if t.exists():
            arr, grid = _band(t, key)
            diag(f"aster_{key}.png", arr, grid, f"ASTER GED - {label}", "index (2-98% stretch)", cmap=cm, background=bg)
    return count
