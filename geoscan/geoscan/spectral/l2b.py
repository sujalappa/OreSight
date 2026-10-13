"""EMIT L2B mineral identification (NASA Tetracorder) on the EMIT analysis grid.

L2B is not in Earth Engine, so this reads local EMIT_L2B_MIN / MINUNCERT NetCDF files
(full granules or AppEEARS orthorectified subsets with 1-D lat/lon).

Each library entry is assigned to a mineral class by its PRIMARY constituent, i.e. the
first mineral in the Tetracorder name ("Calcite+.33Muscov ..." -> carbonate). The full
name -> class table is written to the outputs so the assignment can be audited.

Per class and pixel the outputs are:
    n_dates     number of scenes identifying the class (after fit / uncertainty QC)
    max_depth   maximum band depth reported for the class
    mean_fit    mean Tetracorder fit of those identifications
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import netCDF4
import numpy as np
from affine import Affine
from rasterio.warp import Resampling, reproject

from ..config import L2BConfig
from ..grid import Grid

log = logging.getLogger(__name__)

NON_MINERAL = re.compile(r"plastic|plywood|fiberglass|polystyrene|painted|tarp|vinyl|hdpe|pete|"
                         r"grass|vegetation|lichen|wood|asphalt|concrete|brick|co2|transm|perchlorate", re.I)

# Order matters: first match wins.
CLASSES: tuple[tuple[str, str, str], ...] = (
    ("white_mica", r"muscov|illite|sericite|phengite|paragonite", "Muscovite / sericite / illite"),
    ("kaolinite", r"kaol|dickite|halloysite|nacrite", "Kaolinite group"),
    ("smectite", r"montmor|nontron|beidell|saponite|smectite|hectorite", "Smectite (montmorillonite, nontronite)"),
    ("alunite", r"alunite", "Alunite"),
    ("pyrophyllite", r"pyrophyl", "Pyrophyllite"),
    ("chlorite", r"chlorite|clinochlore|chamosite|thuringite", "Chlorite"),
    ("epidote", r"epidote|clinozoisite|zoisite", "Epidote group"),
    ("prehnite", r"prehnite", "Prehnite"),
    ("amphibole", r"actinolite|tremolite|hornblende|cummingtonite|anthophyllite|riebeckite", "Amphibole"),
    ("serpentine", r"serpentine|antigorite|lizardite|chrysotile|cronstedtite|greenalite", "Serpentine group"),
    ("talc", r"talc", "Talc"),
    ("carbonate", r"calcite|dolomite|ankerite|siderite|magnesite|aragonite|strontianite|witherite|rhodochrosite",
     "Carbonate"),
    ("jarosite", r"jarosite", "Jarosite"),
    ("ferric_oxide", r"hematit|goeth|ferrihydrite|lepidocrocite|maghemite|fe-hydrox|limonite|schwertmannite",
     "Ferric oxide / oxyhydroxide"),
    ("copper_mineral", r"chrysocolla|malachite|azurite|cuprite|chalcanthite|brochantite", "Copper secondary mineral"),
    ("sulfate", r"gypsum|anhydrite|kieserite|epsomite", "Sulfate"),
    ("vermiculite", r"vermiculite", "Vermiculite"),
    ("biotite", r"biotite|phlogopite", "Biotite / phlogopite"),
    ("pyroxene_olivine", r"pyroxene|augite|diopside|enstatite|hypersthene|olivine|fayalite|forsterite",
     "Pyroxene / olivine"),
)
CLASS_KEYS = [c[0] for c in CLASSES] + ["other"]
CLASS_LABELS = {**{c[0]: c[2] for c in CLASSES}, "other": "Other / unassigned"}


def classify_name(name: str) -> str | None:
    """Mineral class of a Tetracorder library entry, or None for non-minerals."""
    if NON_MINERAL.search(name):
        return None
    primary = re.split(r"[+\s]", name.strip(), maxsplit=1)[0]
    for key, pattern, _ in CLASSES:
        if re.search(pattern, primary, re.I):
            return key
    # Primary token not recognised (e.g. an abbreviation): fall back to the full name.
    for key, pattern, _ in CLASSES:
        if re.search(pattern, name, re.I):
            return key
    return "other"


def _names(ds) -> dict[int, str]:
    raw = ds.variables["name"][:]
    idx = ds.variables["index"][:]
    out = {}
    for i, n in zip(np.asarray(idx).ravel(), raw):
        if isinstance(n, (bytes, np.bytes_)):
            n = n.decode("utf-8", "ignore")
        elif not isinstance(n, str):
            n = b"".join(c for c in np.ma.filled(n, b"") if c).decode("utf-8", "ignore")
        out[int(i)] = str(n).strip()
    return out


def _granule_key(path: Path) -> str:
    m = re.search(r"(\d{8}T\d{6}_\d+_\d+)", path.name)
    return m.group(1) if m else path.stem


def _native_transform(lat: np.ndarray, lon: np.ndarray) -> tuple[Affine, bool]:
    """Transform for 1-D regular lat/lon axes. Returns (transform, rows_need_flip)."""
    if lat.ndim != 1 or lon.ndim != 1:
        raise NotImplementedError("Only orthorectified L2B files with 1-D lat/lon are supported")
    dlon = float(np.mean(np.diff(lon)))
    dlat = float(np.mean(np.diff(lat)))
    flip = dlat > 0  # rows ordered south->north: flip so row 0 is north
    res_y = abs(dlat)
    north = float(lat.max()) + res_y / 2
    west = float(lon.min()) - abs(dlon) / 2
    return Affine(abs(dlon), 0, west, 0, -res_y, north), flip


def _to_grid(arr: np.ndarray, src_transform: Affine, grid: Grid) -> np.ndarray:
    dst = np.full(grid.shape, np.nan, dtype="float32")
    reproject(arr.astype("float32"), dst, src_transform=src_transform, src_crs="EPSG:4326",
              dst_transform=grid.transform, dst_crs=grid.crs, resampling=Resampling.nearest,
              src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def find_granules(folder: Path) -> list[tuple[Path, Path]]:
    mins = {_granule_key(p): p for p in Path(folder).glob("*L2B_MIN_*.nc")}
    uncs = {_granule_key(p): p for p in Path(folder).glob("*L2B_MINUNCERT_*.nc")}
    pairs = []
    for key in sorted(mins):
        if key in uncs:
            pairs.append((mins[key], uncs[key]))
        else:
            log.warning("L2B %s has no MINUNCERT companion; skipped", mins[key].name)
    return pairs


def process_l2b(folder: Path, grid: Grid, inside_aoi: np.ndarray, cfg: L2BConfig):
    """Returns (layers dict, per-granule stats list, name->class table rows)."""
    pairs = find_granules(folder)
    n_dates = {k: np.zeros(grid.shape, "float32") for k in CLASS_KEYS}
    max_depth = {k: np.full(grid.shape, np.nan, "float32") for k in CLASS_KEYS}
    fit_sum = {k: np.zeros(grid.shape, "float32") for k in CLASS_KEYS}
    w_sum = {k: np.zeros(grid.shape, "float32") for k in CLASS_KEYS}
    wbd_sum = {k: np.zeros(grid.shape, "float32") for k in CLASS_KEYS}
    observed = np.zeros(grid.shape, bool)
    table: dict[str, dict] = {}
    stats = []

    for f_min, f_unc in pairs:
        with netCDF4.Dataset(f_min) as dm, netCDF4.Dataset(f_unc) as du:
            names = _names(dm)
            lat, lon = np.asarray(dm.variables["lat"][:]), np.asarray(dm.variables["lon"][:])
            transform, flip = _native_transform(lat, lon)

            def read(ds, var, fill):
                a = np.ma.filled(ds.variables[var][:].astype("float32"), fill)
                return a[::-1] if flip else a

            scene_hits = {}
            footprint = None
            for group, min_fit, max_unc in ((1, cfg.group1_min_fit, cfg.group1_max_unc),
                                            (2, cfg.group2_min_fit, cfg.group2_max_unc)):
                mid = read(dm, f"group_{group}_mineral_id", -9999)
                bd = read(dm, f"group_{group}_band_depth", np.nan)
                fit = read(du, f"group_{group}_fit", np.nan)
                unc = read(du, f"group_{group}_band_depth_unc", np.nan)
                fp = _to_grid((mid != -9999).astype("float32"), transform, grid) > 0.5
                footprint = fp if footprint is None else footprint | fp

                ok = (mid > 0) & (bd > 0) & (fit >= min_fit) & (unc <= max_unc)
                cls_native = np.full(mid.shape, -1, dtype="int16")
                for m in np.unique(mid[mid > 0]).astype(int):
                    name = names.get(m, f"id_{m}")
                    key = classify_name(name)
                    row = table.setdefault(name, {"library_name": name, "group": group,
                                                  "class": key or "non_mineral", "pixels_passing_qc": 0})
                    sel = (mid == m) & ok
                    row["pixels_passing_qc"] += int(sel.sum())
                    if key is not None:
                        cls_native[sel] = CLASS_KEYS.index(key)

                cls = _to_grid(cls_native.astype("float32"), transform, grid)
                bd_g = _to_grid(np.where(cls_native >= 0, bd, np.nan), transform, grid)
                fit_g = _to_grid(np.where(cls_native >= 0, fit, np.nan), transform, grid)
                unc_g = _to_grid(np.where(cls_native >= 0, unc, np.nan), transform, grid)
                w_g = fit_g / (unc_g + 1e-4)          # legacy observation weight: fit / uncertainty
                for ci, key in enumerate(CLASS_KEYS):
                    hit = (cls == ci) & inside_aoi
                    if not hit.any():
                        continue
                    scene_hits[key] = scene_hits.get(key, np.zeros(grid.shape, bool)) | hit
                    max_depth[key] = np.where(hit, np.fmax(max_depth[key], bd_g), max_depth[key])
                    fit_sum[key] += np.where(hit, fit_g, 0.0)
                    w_sum[key] += np.where(hit, np.nan_to_num(w_g), 0.0)
                    wbd_sum[key] += np.where(hit, np.nan_to_num(w_g * bd_g), 0.0)

            for key, hit in scene_hits.items():
                n_dates[key] += hit
            observed |= footprint & inside_aoi
            stats.append({"granule": f_min.name, "key": _granule_key(f_min),
                          **{f"px_{k}": int(v.sum()) for k, v in scene_hits.items()}})

    layers = {}
    for key in CLASS_KEYS:
        n = n_dates[key]
        layers[key] = {
            "n_dates": np.where(observed, n, np.nan).astype("float32"),
            "max_depth": max_depth[key],
            "mean_fit": np.where(n > 0, fit_sum[key] / np.maximum(n, 1), np.nan).astype("float32"),
            # multi-date sum of fit/unc weights and the weighted mean band depth (legacy weighting)
            "weight": np.where(observed, w_sum[key], np.nan).astype("float32"),
            "weighted_depth": np.where(w_sum[key] > 0, wbd_sum[key] / np.maximum(w_sum[key], 1e-9),
                                       np.nan).astype("float32"),
        }
    rows = sorted(table.values(), key=lambda r: -r["pixels_passing_qc"])
    return layers, stats, rows, len(pairs)
