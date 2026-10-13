"""Build the reviewer edition of the Phase-1 deliverables (main NDVI <= 0.30 product only).

    python scripts/build_review_package.py [--date 2026-09-24]

Included per block: the 4 alteration maps, CPI target map, lineament map, rose diagram; the
alteration / target / structure / terrain / ASTER rasters; all vectors; the targets, alteration,
lineament and ASTER tables; a filtered manifest; and the validation report.

Excluded (internal only): any earlier-method outputs or comparisons, experiments, pixel-level diagnostics, EMIT L2A/L2B pixel products, scene usability
and vegetation / data-coverage statistics (the data-support band and column, the NDVI band), the
extended-coverage NDVI variants and the NDVI experiment.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from geoscan.raster_io import read_raster, write_raster  # noqa: E402

ROOT = HERE.parents[1]
BLOCKS = {"kehalpur": "02_Kehalpur", "thanewasna": "03_Thanewasna"}
MAPS = ["alteration_muscovite_map.png", "alteration_chlorite_map.png", "alteration_gossan_map.png",
        "alteration_quartz_map.png", "cpi_target_map.png", "structure_lineaments_map.png",
        "structure_rose_diagram.png"]
RASTERS = ["alteration/muscovite.tif", "alteration/chlorite.tif", "alteration/gossan.tif", "alteration/quartz.tif",
           "alteration/alteration_stack_30m.tif", "targets/cpi.tif", "structure/lineament_distance.tif",
           "structure/lineament_density.tif", "terrain/dem.tif", "terrain/hillshade.tif", "terrain/slope.tif",
           "aster_ged/tir_indices.tif"]
DROP_BANDS = {"data_support", "ged_ndvi", "ged_num_obs"}
TABLES = ["top50_targets.csv", "alteration_summary.csv", "lineaments.csv", "lineament_rose.csv",
          "aster_ged_index_summary.csv", "emit_l2b_library_classes.csv", "l2a_vs_l2b_crosscheck.csv"]
DROP_COLUMNS = {"EMIT_Data_Support"}


def newest(path: Path) -> Path:
    alt = path.with_name(path.stem + "_new" + path.suffix)
    return alt if alt.exists() and (not path.exists() or alt.stat().st_mtime > path.stat().st_mtime) else path


def copy_raster(src: Path, dst: Path):
    data, grid, names, tags = read_raster(src)
    keep = [i for i, n in enumerate(names) if n not in DROP_BANDS]
    write_raster(dst, data[keep], grid, [names[i] for i in keep], tags=tags)


def product(block: str, dst: Path):
    src = ROOT / "outputs" / block
    included = []
    for m in MAPS:
        p = newest(src / "maps" / m)
        (dst / "maps").mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / "maps" / m)
        included.append(f"maps/{m}")
    for r in RASTERS:
        if (src / "rasters" / r).exists():
            copy_raster(src / "rasters" / r, dst / "rasters" / r)
            included.append(f"rasters/{r}")
    import geopandas as gpd
    (dst / "vectors").mkdir(parents=True, exist_ok=True)
    for p in sorted((src / "vectors").iterdir()):
        g = gpd.read_file(p)
        g = g.drop(columns=[c for c in g.columns if c in DROP_COLUMNS])
        g.to_file(dst / "vectors" / p.name, driver="GPKG" if p.suffix == ".gpkg" else "GeoJSON")
        included.append(f"vectors/{p.name}")
    (dst / "tables").mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        if (src / "tables" / t).exists():
            df = pd.read_csv(src / "tables" / t)
            df.drop(columns=[c for c in df.columns if c in DROP_COLUMNS]).to_csv(dst / "tables" / t, index=False)
            included.append(f"tables/{t}")
    m = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    outputs = {}
    for k in included:
        if k in m["outputs"]:
            e = dict(m["outputs"][k])
            if "bands" in e:
                e["bands"] = [b for b in e["bands"] if b not in DROP_BANDS]
            outputs[k] = e
    manifest = {"edition": "reviewer edition (main product)", "aoi": m["aoi"], "grids": m["grids"],
                "settings": m["settings"], "steps": {k: v for k, v in m["steps"].items()
                                                     if k in ("structure", "alteration", "targets")},
                "outputs": outputs, "software": m["software"], "updated_utc": m["updated_utc"]}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validation(block: str, dst: Path):
    """New-method validation produced by scripts/validate.py (outputs/<block>_validation)."""
    src = ROOT / "outputs" / f"{block}_validation"
    if not (src / "summary.md").exists():
        raise FileNotFoundError(f"Run: python scripts/validate.py --name {block}")
    shutil.copytree(src, dst)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-09-24")
    args = ap.parse_args(argv)
    name = f"Phase1_geologist_review_{args.date}"
    base = ROOT / "deliverables"
    out = base / name
    if out.exists():
        shutil.rmtree(out)
    for f in (base / f"{name}.zip",):
        f.unlink(missing_ok=True)
    (out / "01_Methodology").mkdir(parents=True)
    for block, folder in BLOCKS.items():
        product(block, out / folder / "A_main_product")
        validation(block, out / folder / "B_validation_vs_ground_truth")
    shutil.copy2(HERE.parent / "review" / "Phase1_Methodology.md", out / "01_Methodology")
    shutil.copy2(HERE.parent / "review" / "00_READ_ME_FIRST.md", out)
    # styled Word documents (reference styles, native equations, TOC refreshed in Word, PDF preview)
    sys.path.insert(0, str(HERE))
    import make_word_docs
    sys.argv = ["make_word_docs.py", str(out)]
    make_word_docs.main()
    # quality gate: refuse to zip a package that fails any raster / vector / consistency check
    import subprocess
    qa = subprocess.run([sys.executable, str(HERE / "qa_package.py"), str(out)])
    if qa.returncode != 0:
        raise SystemExit("QA failed - package NOT zipped. See deliverables/_qa_report.csv")
    shutil.make_archive(str(base / name), "zip", base, name)
    print(out, f"{sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
