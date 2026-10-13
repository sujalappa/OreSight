"""Folder layout for one AOI.

    <root>/data/<aoi>/aoi/                 the AOI polygon as supplied + normalised GeoJSON
    <root>/data/<aoi>/raw/emit_l2a/<scene>/ reflectance.tif, uncertainty.tif, qa.tif, meta.json
    <root>/data/<aoi>/raw/emit_l2b/         EMIT L2B MIN + MINUNCERT NetCDF (AppEEARS subsets)
    <root>/data/<aoi>/raw/srtm/             dem_30m.tif (padded)
    <root>/data/<aoi>/raw/aster_ged/        ged_90m.tif
    <root>/data/<aoi>/processed/            per-scene intermediate products
    <root>/outputs/<aoi>/                   final deliverables only (rasters, vectors, tables, maps, manifest)

Raw downloads are cached and never overwritten; outputs are rebuilt on every full run.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Workspace:
    root: Path
    aoi_name: str
    output_name: str | None = None   # variant runs share data/<aoi> but write outputs/<output_name>

    def __post_init__(self):
        self.root = Path(self.root)
        self.output_name = self.output_name or self.aoi_name

    @property
    def data(self) -> Path:
        return self.root / "data" / self.aoi_name

    @property
    def aoi_dir(self) -> Path:
        return self.data / "aoi"

    def raw(self, source: str) -> Path:
        return self.data / "raw" / source

    def processed(self, step: str) -> Path:
        base = self.data / "processed"
        return base / step if self.output_name == self.aoi_name else base / self.output_name / step

    @property
    def outputs(self) -> Path:
        return self.root / "outputs" / self.output_name

    def out(self, *parts: str) -> Path:
        p = self.outputs.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def clean_outputs(self) -> None:
        """Delete everything inside <root>/outputs/<aoi> (and nothing outside it).

        Folders held open by another program (e.g. File Explorer on Windows) may survive
        as empty directories; any surviving *file* is an error, because it would be stale.
        """
        assert self.outputs.parent == self.root / "outputs"
        if self.outputs.exists():
            for child in self.outputs.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            leftovers = [p for p in self.outputs.rglob("*") if p.is_file()]
            if leftovers:
                raise PermissionError(f"Could not delete old outputs (open in another program?): {leftovers[:3]}")
        self.outputs.mkdir(parents=True, exist_ok=True)
