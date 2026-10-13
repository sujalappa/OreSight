"""Produce the extended-coverage variants (vegetation mask NDVI <= 0.35 and <= 0.40).

    python scripts/run_ndvi_variants.py

Each variant re-uses the cached raw data in data/<block>/ and is written to its own folder,
outputs/<block>_ndvi035 and outputs/<block>_ndvi040, with every map labelled as extended
coverage / lower confidence. The main products (NDVI <= 0.30) in outputs/<block>/ are not touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from geoscan.config import load_settings  # noqa: E402
from geoscan.pipeline import Pipeline  # noqa: E402

ROOT = HERE.parents[1]
CONFIG = HERE.parent / "config" / "default.yaml"
AOIS = {"kehalpur": ROOT / "data/kehalpur/aoi/kehalpur_Cu_Known_Block.geojson",
        "thanewasna": ROOT / "data/thanewasna/aoi/thanewasna_emit_download_area.geojson"}
VARIANTS = {"ndvi035": 0.35, "ndvi040": 0.40}
STEPS = ("emit_l2a", "emit_l2b", "aster", "structure", "alteration", "targets", "maps")


def main():
    for block, aoi in AOIS.items():
        for suffix, ndvi in VARIANTS.items():
            label = f"EXTENDED COVERAGE NDVI<={ndvi:.2f} - lower confidence"
            s = load_settings(CONFIG, {"spectral": {"ndvi_max": ndvi}, "run_label": label})
            p = Pipeline(aoi, block, ROOT, s, output_name=f"{block}_{suffix}")
            p.run(STEPS, clean=True)
            print(f"[done] {block} {suffix} -> {p.ws.outputs}")


if __name__ == "__main__":
    main()
