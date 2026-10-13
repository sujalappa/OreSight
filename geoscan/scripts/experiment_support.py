"""Experiment: how locally should "EMIT data support" (assessed / not assessed) be judged?

    python scripts/experiment_support.py

Support only decides where targets may be placed and where maps are hatched; the mineral layers and
the CPI do not use it. Variants (copies of the main product, alteration -> targets -> maps re-run):
  current     Gaussian sigma = halo/2 (~190 m), support >= 0.20
  local90     Gaussian sigma = 90 m, support >= 0.20
  neighbour   the pixel or one of its 8 neighbours (60 m) has a usable EMIT view
Each is validated with scripts/validate.py; outputs/<block>_support_<variant>[_validation].
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from geoscan.config import load_settings  # noqa: E402
from geoscan.pipeline import Pipeline  # noqa: E402
from geoscan.raster_io import read_raster  # noqa: E402

ROOT = HERE.parents[1]
CONFIG = HERE.parent / "config" / "default.yaml"
VARIANTS = {"local90": 90.0, "neighbour": 0.0}


def run(block: str, name: str, sigma):
    out = ROOT / "outputs" / f"{block}_support_{name}"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "outputs" / block, out, ignore=shutil.ignore_patterns("_webapp_cache", "maps"))
    s = load_settings(CONFIG, {"alteration": {"support_sigma_m": sigma}, "run_label": f"support test {name}"})
    p = Pipeline(ROOT / "data" / block / "aoi" / "aoi.geojson", block, ROOT, s, progress=lambda a, b: None,
                 output_name=out.name)
    p.run(("alteration", "targets", "maps"), clean=False)
    subprocess.run([sys.executable, str(HERE / "validate.py"), "--name", block, "--output", out.name],
                   check=True, capture_output=True)
    return out


def summary(block: str, folder: str, label: str, main_targets: pd.DataFrame) -> dict:
    out = ROOT / "outputs" / folder
    val = ROOT / "outputs" / f"{folder}_validation"
    t = pd.read_csv(out / "tables/top50_targets.csv")
    st, g, names, _ = read_raster(out / "rasters/alteration/alteration_stack_30m.tif")
    sup = st[names.index("data_support")]
    inside = np.isfinite(sup)
    row = {"block": block, "variant": label, "targets": len(t),
           "pct_block_eligible": round(100 * float((sup[inside] >= 0.2).mean()), 1)}
    d = np.hypot(t.Easting.values[:, None] - main_targets.Easting.values[None],
                 t.Northing.values[:, None] - main_targets.Northing.values[None]).min(1)
    row["same_as_main_within_100m"] = int((d <= 100).sum())
    if block == "kehalpur":
        b = pd.read_csv(val / "tables/ngdr_exploration_blocks.csv").set_index("report")
        for rid, short in (("CRO-20859-1986", "bhawratekra"), ("CRO-23431-2018", "banskhapa_G2"),
                           ("CRO-22666-2009", "banskhapa_minzone"), ("CRO-21815-1993", "bargaon")):
            row[f"in_{short}"] = int(b.loc[rid, "targets_inside"])
            row[f"exp_{short}"] = b.loc[rid, "targets_expected_at_random"]
        lt = pd.read_csv(val / "tables/targets_by_lithology.csv")
        row["on_cover"] = int(lt.loc[lt.lithology_group.str.contains("Deccan"), "targets"].iloc[0])
    else:
        h = pd.read_csv(val / "tables/target_hits.csv").set_index("subset")
        row["top10_within_100m"] = int(h.loc["top 10", "hits_within_100m"])
        row["all_within_100m"] = int(h.loc["all", "hits_within_100m"])
        row["all_expected_100m"] = h.loc["all", "expected_random_100m"]
    return row


def main():
    rows = []
    for block in ("kehalpur", "thanewasna"):
        main_t = pd.read_csv(ROOT / "outputs" / block / "tables/top50_targets.csv")
        rows.append(summary(block, block, "current (~190 m)", main_t))
        for name, sigma in VARIANTS.items():
            run(block, name, sigma)
            rows.append(summary(block, f"{block}_support_{name}", name, main_t))
            print("done", block, name, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "support_experiment.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.to_string())


if __name__ == "__main__":
    main()
