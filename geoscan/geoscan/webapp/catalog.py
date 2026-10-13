"""Which areas and results exist, and which method options / parameters each result was made with.

A *block* is an area with data in ``data/<block>/``. A *run* is a results folder (any folder
under ``outputs/`` with a geoscan ``manifest.json``) whose settings equal the default settings
except for the user-selectable OPTIONS and PARAMS below. Runs made with any other change
(experiments, earlier-method comparisons) are not listed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, load_settings
from ..pipeline import STEPS

# workspace with data/ and outputs/; GEOSCAN_ROOT overrides it on a server
ROOT = Path(os.environ.get("GEOSCAN_ROOT") or Path(__file__).resolve().parents[3])
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

# Method options (choice lists). "public": False keeps an option out of the interface.
OPTIONS = {
    "structure.dem_source": {
        "label": "Elevation model for lineaments", "step": "structure", "public": True,
        "choices": {"aw3d30": "ALOS AW3D30", "srtm": "SRTM", "copernicus": "Copernicus GLO-30"}},
    "structure.canny_tile_m": {
        "label": "Lineament edge thresholds", "step": "structure", "public": True,
        "choices": {3000.0: "Per 3 km tile", 0.0: "Whole area"}},
    "spectral.ndvi_max": {
        "label": "Vegetation mask (NDVI ≤)", "step": "emit_l2a", "public": False,
        "choices": {0.30: "0.30", 0.35: "0.35", 0.40: "0.40"}},
}


def _p(key, label, group, step, tag, lo, hi, inc, unit="", help_="", kind="float", public=True):
    return key, {"label": label, "group": group, "step": step, "tag": tag, "min": lo, "max": hi, "step_size": inc,
                 "unit": unit, "help": help_, "kind": kind, "public": public}


# Adjustable parameters (numbers the method chose; tags as in the methodology).
PARAMS = dict([
    _p("targets.w_muscovite", "Weight: muscovite", "CPI, zones and field-check locations", "targets", "EXP", 0, 1, 0.01,
       help_="H = w_musc·musc + w_qtz·qtz + w_goss·goss + w_chl·chl. The four weights must add up to 1."),
    _p("targets.w_quartz", "Weight: quartz", "CPI, zones and field-check locations", "targets", "EXP", 0, 1, 0.01),
    _p("targets.w_gossan", "Weight: gossan", "CPI, zones and field-check locations", "targets", "EXP", 0, 1, 0.01),
    _p("targets.w_chlorite", "Weight: chlorite", "CPI, zones and field-check locations", "targets", "EXP", 0, 1, 0.01),
    _p("targets.w_structure", "Structure share of CPI", "CPI, zones and field-check locations", "targets", "EXP", 0, 1, 0.01,
       help_="CPI = (1 − s)·H + s·exp(−d/λ). s = 0.30 by default."),
    _p("targets.structure_lambda_m", "Structural decay distance λ", "CPI, zones and field-check locations", "targets", "EXP",
       50, 2000, 10, "m", "Distance over which the structural score falls to 37 %."),
    _p("targets.tier1", "Tier 1 threshold (CPI)", "CPI, zones and field-check locations", "targets", "EXP", 0, 1, 0.01),
    _p("targets.tier2", "Tier 2 threshold = zone boundary (CPI)", "CPI, zones and field-check locations", "targets", "EXP", 0, 1, 0.01),
    _p("targets.tier3", "Tier 3 threshold (CPI)", "CPI, zones and field-check locations", "targets", "EXP", 0, 1, 0.01),
    _p("targets.min_cpi", "Minimum CPI for a field-check location", "CPI, zones and field-check locations", "targets", "EXP", 0, 1, 0.01),
    _p("targets.min_spacing_m", "Minimum spacing between field-check locations", "CPI, zones and field-check locations", "targets", "ASM",
       30, 2000, 10, "m"),
    _p("targets.n_targets", "Maximum number of field-check locations", "CPI, zones and field-check locations", "targets", "ASM", 1, 200, 1,
       kind="int"),
    _p("alteration.sigma_core_m", "Core kernel σ", "Alteration layers", "alteration", "EXP", 30, 500, 5, "m",
       "Gaussian kernel for the vent/core part of the potential field."),
    _p("alteration.sigma_halo_m", "Halo kernel σ", "Alteration layers", "alteration", "EXP", 60, 2000, 5, "m",
       "Gaussian kernel for the hydrothermal halo. Must be larger than the core σ."),
    _p("alteration.theta_decay", "Halo confidence θ", "Alteration layers", "alteration", "EXP", 0.005, 1, 0.005,
       help_="confidence = 1 − exp(−W_halo / θ)"),
    _p("alteration.reference_percentile", "Reference percentile (100 %)", "Alteration layers", "alteration", "ASM",
       80, 100, 0.5, "", "The field value at this percentile is shown as 100 %."),
    _p("alteration.sericite_min_nm", "White-mica Al-OH position: from", "Alteration layers", "alteration", "LIB",
       2170, 2215, 1, "nm"),
    _p("alteration.sericite_max_nm", "White-mica Al-OH position: to", "Alteration layers", "alteration", "LIB",
       2195, 2230, 1, "nm"),
    _p("alteration.quartz_qi_weight", "Quartz: weight of QI in QSI", "Alteration layers", "alteration", "EXP",
       0, 1, 0.01, help_="QSI = w·QI + (1 − w)·SI"),
    _p("alteration.quartz_background_percentile", "Quartz background percentile", "Alteration layers", "alteration",
       "ASM", 0, 50, 1),
    _p("alteration.min_support", "Minimum EMIT data support", "Alteration layers", "alteration", "ASM", 0, 1, 0.05,
       public=False),
    _p("structure.min_length_m", "Minimum lineament length", "Lineaments", "structure", "EXP", 200, 5000, 50, "m"),
    _p("structure.min_segment_m", "Minimum Hough segment", "Lineaments", "structure", "ASM", 100, 2000, 10, "m"),
    _p("structure.max_gap_m", "Maximum gap within a segment", "Lineaments", "structure", "ASM", 0, 500, 10, "m"),
    _p("structure.canny_low_quantile", "Edge threshold: low quantile", "Lineaments", "structure", "ASM",
       0.5, 0.98, 0.01),
    _p("structure.canny_high_quantile", "Edge threshold: high quantile", "Lineaments", "structure", "ASM",
       0.6, 0.995, 0.005),
    _p("structure.scarp_min_step_m", "Scarp: minimum one-sided step", "Lineaments", "structure", "EXP",
       0.5, 20, 0.5, "m"),
    _p("structure.exclude_scarps", "Leave scarps out of the CPI", "Lineaments", "structure", "ASM", 0, 1, 1,
       kind="bool"),
    _p("spectral.detect_sigma", "Detection: depth > k·σ, k =", "EMIT detection (slow to re-run)", "emit_l2a", "ASM",
       1, 5, 0.1),
    _p("spectral.min_depth", "Detection: minimum band depth", "EMIT detection (slow to re-run)", "emit_l2a", "ASM",
       0, 0.1, 0.001),
    _p("spectral.min_albedo", "Mask pixels darker than (albedo)", "EMIT detection (slow to re-run)", "emit_l2a", "ASM",
       0, 0.2, 0.005),
])
DIMS = {**OPTIONS, **PARAMS}
# settings covered by the pseudo-parameters above (derived, so not compared directly)
DERIVED = {"targets.w_hydrothermal", "alteration.quartz_weights", "alteration.sericite_position_nm"}
IGNORED = ("acquire.", "run_label")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = json.loads(json.dumps(v))    # tuples -> lists, as in a manifest
    return out


def _norm(v):
    if isinstance(v, bool):
        return v
    return round(float(v), 6) if isinstance(v, (int, float)) else v


def values_of(flat: dict) -> dict:
    """All option / parameter values from flattened settings (pseudo keys resolved)."""
    v = {}
    for k in DIMS:
        if k == "alteration.quartz_qi_weight":
            v[k] = _norm(flat["alteration.quartz_weights"][0])
        elif k == "alteration.sericite_min_nm":
            v[k] = _norm(flat["alteration.sericite_position_nm"][0])
        elif k == "alteration.sericite_max_nm":
            v[k] = _norm(flat["alteration.sericite_position_nm"][1])
        elif PARAMS.get(k, {}).get("kind") == "int":
            v[k] = int(flat[k])
        else:
            v[k] = _norm(flat[k])
    return v


def default_settings() -> Settings:
    return load_settings(CONFIG)


def default_values() -> dict:
    return values_of(_flatten(default_settings().to_dict()))


def default_options() -> dict:
    d = default_values()
    return {k: d[k] for k in OPTIONS}


def public_options() -> dict:
    d = default_values()
    from .help_text import INFO
    return {k: {"label": o["label"], "default": d[k], "step": o["step"], "info": INFO.get(k, {}),
                "choices": [{"value": _norm(v), "label": lab} for v, lab in o["choices"].items()]}
            for k, o in OPTIONS.items() if o["public"]}


def public_params() -> dict:
    d = default_values()
    from .help_text import INFO
    return {k: {**{x: y for x, y in p.items() if x != "public"}, "default": d[k], "info": INFO.get(k, {})}
            for k, p in PARAMS.items() if p["public"]}


def changed(values: dict) -> dict:
    d = default_values()
    return {k: v for k, v in values.items() if k in PARAMS and v != d[k]}


def option_label(values: dict) -> str:
    parts = [f"{o['label']}: {o['choices'].get(values[k], values[k])}" for k, o in OPTIONS.items() if o["public"]]
    ch = changed(values)
    if ch:
        parts.append("Custom parameters: " + ", ".join(
            f"{PARAMS[k]['label']} = {v}{(' ' + PARAMS[k]['unit']) if PARAMS[k]['unit'] else ''}" for k, v in ch.items()))
    return " · ".join(parts)


def settings_for(values: dict) -> Settings:
    over: dict = {}

    def put(key, val):
        sect, name = key.split(".")
        over.setdefault(sect, {})[name] = val

    v = {**default_values(), **values}
    for k, val in v.items():
        if k == "alteration.quartz_qi_weight":
            put("alteration.quartz_weights", (val, round(1 - val, 6)))
        elif k in ("alteration.sericite_min_nm", "alteration.sericite_max_nm"):
            put("alteration.sericite_position_nm", (v["alteration.sericite_min_nm"], v["alteration.sericite_max_nm"]))
        elif k == "targets.w_structure":
            put(k, val)
            put("targets.w_hydrothermal", round(1 - val, 6))
        else:
            put(k, val)
    return load_settings(CONFIG, over)


def first_step(values: dict) -> str:
    """Earliest pipeline step affected by any non-default value ('maps' if none)."""
    d = default_values()
    steps = [DIMS[k]["step"] for k, v in values.items() if v != d[k]]
    return min(steps, key=STEPS.index) if steps else "maps"


def variant_id(values: dict) -> str:
    """Folder suffix, e.g. 'dem-srtm_tile-0' or 'dem-srtm_p3f9a21c0'."""
    short = {"structure.dem_source": "dem", "structure.canny_tile_m": "tile", "spectral.ndvi_max": "ndvi"}
    d = default_values()
    parts = [f"{short[k]}-{values[k]:g}" if isinstance(values[k], float) else f"{short[k]}-{values[k]}"
             for k in OPTIONS if values[k] != d[k]]
    ch = changed(values)
    if ch:
        parts.append("p" + hashlib.sha1(json.dumps(ch, sort_keys=True).encode()).hexdigest()[:8])
    return "_".join(parts)


@dataclass
class Run:
    run_id: str              # path under outputs/, '/' replaced by '~' (URL safe)
    path: Path
    block: str
    values: dict
    recommended: bool
    updated: str
    validation: Path | None = None
    extra: dict = field(default_factory=dict)

    @property
    def options(self) -> dict:
        return {k: self.values[k] for k in OPTIONS}

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "block": self.block,
                "options": {k: self.values[k] for k, o in OPTIONS.items() if o["public"]},
                "params": {k: self.values[k] for k, p in PARAMS.items() if p["public"]},
                "custom": bool(changed(self.values)), "label": option_label(self.values),
                "recommended": self.recommended, "updated": self.updated,
                "has_validation": self.validation is not None, **self.extra}


def _validation_dir(run_dir: Path, block: str) -> Path | None:
    cands = [run_dir.parent / f"{run_dir.name}_validation"]
    if run_dir.name.startswith(block):
        cands.append(run_dir.parent / run_dir.name.replace(block, f"{block}_validation", 1))
    return next((c for c in cands if (c / "summary.md").exists()), None)


def runs() -> list[Run]:
    base = _flatten(Settings().to_dict())              # code defaults fill keys missing in older manifests
    ref = _flatten(default_settings().to_dict())
    dflt = default_values()
    found = []
    for mf in sorted(OUTPUTS.glob("*/manifest.json")) + sorted(OUTPUTS.glob("_archive/*/manifest.json")):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            s = {**base, **_flatten(m["settings"])}
            block = m["aoi"]["name"]
        except Exception:                                # not a geoscan manifest
            continue
        if not (DATA / block).exists() or "outputs" not in m or "targets" not in m.get("steps", {}):
            continue
        same = all(_norm(s.get(k)) == _norm(v) if not isinstance(v, list) else s.get(k) == v
                   for k, v in ref.items()
                   if k not in DIMS and k not in DERIVED and not k.startswith(IGNORED))
        if not same:
            continue
        vals = values_of(s)
        if any(not o["public"] and vals[k] != dflt[k] for k, o in DIMS.items()):
            continue
        rel = mf.parent.relative_to(OUTPUTS).as_posix()
        try:
            by = json.loads((mf.parent / "created_by.json").read_text(encoding="utf-8")).get("user", "")
        except Exception:
            by = ""
        found.append(Run(rel.replace("/", "~"), mf.parent, block, vals, vals == dflt,
                         m.get("updated_utc", ""), _validation_dir(mf.parent, block), {"created_by": by}))
    best: dict = {}
    for r in found:          # one run per (block, values): prefer the main folder, then the newest
        key = (r.block, json.dumps(r.values, sort_keys=True))
        cur = best.get(key)
        if cur is None or (r.path.name == r.block) or (cur.path.name != cur.block and r.updated > cur.updated):
            best[key] = r
    return sorted(best.values(), key=lambda r: (r.block, not r.recommended, bool(changed(r.values)), r.run_id))


def get_run(run_id: str) -> Run:
    for r in runs():
        if r.run_id == run_id:
            return r
    raise KeyError(run_id)


def find_run(block: str, values: dict) -> Run | None:
    values = {k: _norm(v) if not isinstance(v, bool) else v for k, v in values.items()}
    return next((r for r in runs() if r.block == block and r.values == values), None)


def base_run(block: str, values: dict) -> Run | None:
    """An existing run whose inputs are identical up to the first step these values change."""
    start = STEPS.index(first_step(values))
    for r in runs():
        if r.block != block or r.path.parent != OUTPUTS:
            continue
        if all(r.values[k] == values[k] for k, d in DIMS.items() if STEPS.index(d["step"]) < start):
            return r
    return None


def blocks() -> list[dict]:
    from ..aoi import load_aoi
    by_block: dict = {}
    for r in runs():
        by_block.setdefault(r.block, []).append(r.to_dict())
    out = []
    for d in sorted(p for p in DATA.iterdir() if (p / "aoi" / "aoi.geojson").exists()):
        try:
            aoi = load_aoi(d / "aoi" / "aoi.geojson", d.name)
            area, bounds = round(aoi.area_km2(), 2), list(aoi.bounds)
        except Exception:
            area, bounds = None, None
        gt = d / "ground_truth"
        try:
            by = json.loads((d / "aoi" / "uploaded_by.json").read_text(encoding="utf-8")).get("user", "")
        except Exception:
            by = ""
        out.append({"block": d.name, "area_km2": area, "bounds": bounds,
                    "has_ground_truth": gt.exists() and any(gt.iterdir()), "uploaded_by": by,
                    "runs": by_block.get(d.name, [])})
    return out


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s[:40] or "area"
