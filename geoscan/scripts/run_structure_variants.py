"""Structure-layer candidates, run end to end and validated against the main product.

    python scripts/run_structure_variants.py

Variants (written to outputs/<block>_<suffix>; outputs/<block> is not touched):
  lin_srtm_tiled    SRTM, Canny quantile thresholds per 3 km tile
  lin_aw3d30_tiled  ALOS AW3D30 v4.1, Canny quantile thresholds per 3 km tile
Each variant is then scored with scripts/validate.py --output <variant>.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from geoscan.acquire import gee  # noqa: E402
from geoscan.config import load_settings  # noqa: E402
from geoscan.pipeline import Pipeline  # noqa: E402

ROOT = HERE.parents[1]
CONFIG = HERE.parent / "config" / "default.yaml"
AOIS = {"thanewasna": ROOT / "data/thanewasna/aoi/thanewasna_emit_download_area.geojson",
        "kehalpur": ROOT / "data/kehalpur/aoi/kehalpur_Cu_Known_Block.geojson"}
VARIANTS = {"lin_srtm_tiled": {"dem_source": "srtm", "canny_tile_m": 3000.0},
            "lin_aw3d30_tiled": {"dem_source": "aw3d30", "canny_tile_m": 3000.0}}
STEPS = ("emit_l2a", "emit_l2b", "aster", "structure", "alteration", "targets", "maps")


def main():
    for block, aoi in AOIS.items():
        for suffix, structure in VARIANTS.items():
            s = load_settings(CONFIG, {"structure": structure, "run_label": f"structure candidate {suffix}"})
            p = Pipeline(aoi, block, ROOT, s, output_name=f"{block}_{suffix}")
            gee.initialize(s.acquire.ee_project)
            gee.download_dem(structure["dem_source"], p.grid_dem,
                             gee.dem_path(p.ws.raw(""), structure["dem_source"]),
                             s.acquire.max_request_mb, s.acquire.retries)
            p.run(STEPS, clean=True)
            print(f"[done] {block} {suffix} -> {p.ws.outputs}", flush=True)
            subprocess.run([sys.executable, str(HERE / "validate.py"), "--name", block,
                            "--output", f"{block}_{suffix}"], check=True)


if __name__ == "__main__":
    main()
