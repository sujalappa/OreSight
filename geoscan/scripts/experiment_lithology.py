"""Experiment: does lithology background removal improve the alteration layers / CPI?

    python scripts/experiment_lithology.py

Compares alteration.lithology_background = False (baseline) vs True on Thanewasna and Kehalpur
with the scoring in experiment_l2b.py. The fair tests are those independent of the lithology
map: Thanewasna lodes / boreholes and Kehalpur NGDR exploration-block footprints. Kehalpur's
basement-vs-cover and host-vs-non-host contrasts are reported for information only: removing
per-unit backgrounds changes them by construction.

Writes outputs/lithology_experiment/results.csv and leaves both blocks on the baseline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_l2b import AOIS, CONFIG, ROOT, score_kehalpur, score_thanewasna
from geoscan.config import load_settings
from geoscan.pipeline import Pipeline

VARIANTS = {"baseline": {}, "lithology_background": {"alteration": {"lithology_background": True}}}


def run(name, overrides):
    Pipeline(AOIS[name], name, ROOT, load_settings(CONFIG, overrides)).run(("alteration", "targets"), clean=False)


def main():
    out = ROOT / "outputs/lithology_experiment"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for v, ov in VARIANTS.items():
        for name in AOIS:
            run(name, ov)
        rows.append({"variant": v, **score_thanewasna(), **score_kehalpur()})
        if v != "baseline":
            for name in AOIS:
                src = ROOT / "outputs" / name / "tables/lithology_backgrounds.csv"
                if src.exists():
                    pd.read_csv(src).to_csv(out / f"{name}_lithology_backgrounds.csv", index=False)
        print(f"[done] {v}")
    df = pd.DataFrame(rows).set_index("variant").T
    df.to_csv(out / "results.csv")
    pd.set_option("display.width", 200)
    print(df.to_string())
    for name in AOIS:
        run(name, {})


if __name__ == "__main__":
    main()
