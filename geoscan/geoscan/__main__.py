"""Command line interface.

    python -m geoscan scenes --aoi block.kml --ee-project my-project
    python -m geoscan run    --aoi block.kml --name kehalpur --ee-project my-project
    python -m geoscan run    --aoi block.kml --name kehalpur --steps structure,maps
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .aoi import load_aoi
from .config import load_settings
from .pipeline import STEPS, Pipeline

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="geoscan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--aoi", required=True, help="KML, KMZ, GeoJSON or Shapefile polygon")
        p.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML settings file")
        p.add_argument("--ee-project", help="Google Earth Engine cloud project id")

    r = sub.add_parser("run", help="run the pipeline")
    common(r)
    r.add_argument("--name", help="AOI name used for folders (default: file name)")
    r.add_argument("--root", default=str(DEFAULT_ROOT), help="workspace root containing data/ and outputs/")
    r.add_argument("--steps", default=",".join(STEPS), help=f"comma list from {','.join(STEPS)}")
    r.add_argument("--no-clean", action="store_true", help="keep existing outputs on a full run")

    s = sub.add_parser("scenes", help="list EMIT L2A scenes over the AOI")
    common(s)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    overrides = {"acquire": {"ee_project": args.ee_project}} if args.ee_project else None
    settings = load_settings(args.config, overrides)

    if args.cmd == "scenes":
        from .acquire import gee
        gee.initialize(settings.acquire.ee_project)
        for sc in gee.list_emit_scenes(load_aoi(args.aoi), settings.acquire.start_date, settings.acquire.end_date):
            print(f"{sc['datetime']}  coverage={sc['aoi_coverage']:.2f}  {sc['scene_id']}")
        return 0

    steps = tuple(s.strip() for s in args.steps.split(",") if s.strip())
    pipe = Pipeline(args.aoi, args.name or Path(args.aoi).stem, args.root, settings)
    manifest = pipe.run(steps, clean=False if args.no_clean else None)
    print(f"\nOutputs: {pipe.ws.outputs}  ({len(manifest['outputs'])} files, see manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
