"""Experiment: does a looser vegetation mask (spectral.ndvi_max) improve results?

    python scripts/experiment_ndvi.py

Re-runs emit_l2a -> alteration -> targets for Thanewasna and Kehalpur with ndvi_max in
{0.30 (baseline), 0.35, 0.40}, scores each with the tests in experiment_l2b.py, and reports the
share of each block with usable EMIT data. Leaves both blocks on the baseline.
Writes outputs/ndvi_experiment/results.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiment_l2b import AOIS, CONFIG, ROOT, score_kehalpur, score_thanewasna
from geoscan.config import load_settings
from geoscan.pipeline import Pipeline
from geoscan.raster_io import read_raster

VARIANTS = (0.30, 0.35, 0.40)


def run(name, ndvi_max):
    s = load_settings(CONFIG, {"spectral": {"ndvi_max": ndvi_max}})
    Pipeline(AOIS[name], name, ROOT, s).run(("emit_l2a", "alteration", "targets"), clean=False)


def coverage(name):
    q, _, qn, _ = read_raster(ROOT / "outputs" / name / "rasters/emit_l2a/quality_vegetation.tif")
    nv = q[qn.index("n_valid_scenes")]
    inside = np.isfinite(nv)
    stack, _, sn, _ = read_raster(ROOT / "outputs" / name / "rasters/alteration/alteration_stack_30m.tif")
    sup = stack[sn.index("data_support")]
    return {f"{name}_usable_pixels_pct": round(100 * float((nv[inside] > 0).mean()), 1),
            f"{name}_supported_area_pct": round(100 * float((sup[np.isfinite(sup)] >= 0.2).mean()), 1)}


def main():
    out = ROOT / "outputs/ndvi_experiment"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in VARIANTS:
        cov = {}
        for name in AOIS:
            run(name, v)
            cov.update(coverage(name))
        rows.append({"ndvi_max": v, **cov, **score_thanewasna(), **score_kehalpur()})
        print(f"[done] ndvi_max={v}")
    df = pd.DataFrame(rows).set_index("ndvi_max").T
    df.to_csv(out / "results.csv")
    pd.set_option("display.width", 200)
    print(df.to_string())
    for name in AOIS:
        run(name, 0.30)


if __name__ == "__main__":
    main()
