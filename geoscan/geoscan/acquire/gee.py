"""Google Earth Engine acquisition.

Pixels are pulled with ``ee.data.computePixels`` directly onto a geoscan :class:`Grid`,
so Earth Engine does the reprojection once and every downloaded array is already
north-up and aligned with the analysis grid. Large requests are split into spatial
tiles and band chunks to stay under the request payload limit.

Datasets
--------
EMIT L2A  NASA/EMIT/L2A/RFL          reflectance_0..284, reflectance_uncertainity_0..284, masks
SRTM      USGS/SRTMGL1_003           elevation (m)
ASTER GED NASA/ASTER_GED/AG100_003   emissivity_band10..14 (x0.001), emissivity_sdev_* (x0.0001), ndvi (x0.01)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from pathlib import Path

import numpy as np

from ..aoi import AOI
from ..grid import Grid
from ..raster_io import write_raster

log = logging.getLogger(__name__)

EMIT_L2A = "NASA/EMIT/L2A/RFL"
SRTM = "USGS/SRTMGL1_003"
ASTER_GED = "NASA/ASTER_GED/AG100_003"
NODATA = -9999.0
MAX_BANDS_PER_REQUEST = 64

EMIT_QA_BANDS = ["elev", "cloud_flag", "cirrus_flag", "water_flag", "spacecraft_flag",
                 "dilated_cloud_flag", "AOD550", "H2O", "aggregate_flag"]
GED_BANDS = [f"emissivity_band{b}" for b in range(10, 15)] + \
            [f"emissivity_sdev_band{b}" for b in range(10, 15)] + ["ndvi", "land_water_map", "num_obs"]
GED_SCALE = {**{f"emissivity_band{b}": 0.001 for b in range(10, 15)},
             **{f"emissivity_sdev_band{b}": 0.0001 for b in range(10, 15)}, "ndvi": 0.01}


def _ee():
    import ee  # imported lazily so the rest of geoscan works without Earth Engine
    return ee


def initialize(project: str) -> None:
    if not project:
        raise ValueError("Earth Engine project id is required (settings.acquire.ee_project or --ee-project)")
    ee = _ee()
    key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")       # JSON key of a service account (servers)
    if key:
        info = json.loads(key)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=key)
        ee.Initialize(credentials=creds, project=os.environ.get("GEE_PROJECT") or info.get("project_id") or project)
    else:                                                  # personal login (earthengine authenticate)
        ee.Initialize(project=project)


def _aoi_geometry(aoi: AOI):
    return _ee().Geometry(json.loads(json.dumps(aoi.geometry.__geo_interface__)))


def fetch(image, bands: list[str], grid: Grid, max_request_mb: float = 30.0, retries: int = 4) -> np.ndarray:
    """Download ``bands`` of ``image`` onto ``grid``. Returns float32 (bands, rows, cols), NaN = no data."""
    ee = _ee()
    img = image.select(bands).toFloat().unmask(NODATA, False)
    max_bytes = max_request_mb * 1e6
    tile = 512
    out = np.full((len(bands), grid.height, grid.width), np.nan, dtype="float32")

    for r0 in range(0, grid.height, tile):
        for c0 in range(0, grid.width, tile):
            h, w = min(tile, grid.height - r0), min(tile, grid.width - c0)
            sub = grid.subgrid(r0, c0, h, w)
            # numpy refuses .npy headers > 10 kB, which ~100+ named bands exceed; cap bands per request.
            per_chunk = max(1, min(MAX_BANDS_PER_REQUEST, int(max_bytes // (h * w * 4))))
            for b0 in range(0, len(bands), per_chunk):
                chunk = bands[b0:b0 + per_chunk]
                request = {"expression": img.select(chunk), "fileFormat": "NUMPY_NDARRAY", "grid": sub.ee_grid()}
                for attempt in range(retries + 1):
                    try:
                        arr = ee.data.computePixels(request)
                        break
                    except Exception as exc:  # EE raises generic exceptions for quota / transient errors
                        if attempt == retries:
                            raise
                        wait = 2 ** attempt * 5
                        log.warning("computePixels failed (%s); retrying in %ss", str(exc)[:120], wait)
                        time.sleep(wait)
                block = np.stack([arr[name] for name in chunk]).astype("float32")
                block[block == NODATA] = np.nan
                out[b0:b0 + len(chunk), r0:r0 + h, c0:c0 + w] = block
    return out


# ---------------------------------------------------------------- EMIT L2A

def list_emit_scenes(aoi: AOI, start: str, end: str | None = None) -> list[dict]:
    """EMIT L2A scenes intersecting the AOI, with the fraction of the AOI each one covers."""
    ee = _ee()
    geom = _aoi_geometry(aoi)
    col = ee.ImageCollection(EMIT_L2A).filterBounds(geom).filterDate(start, end or date.today().isoformat())
    aoi_area = geom.area(30)

    def describe(img):
        return ee.Feature(None, {
            "scene_id": img.id(),
            "datetime": img.date().format("YYYY-MM-dd'T'HH:mm:ss"),
            "aoi_coverage": img.geometry().intersection(geom, 30).area(30).divide(aoi_area),
            "solar_zenith": img.get("SOLAR_ZENITH"),
            "solar_azimuth": img.get("SOLAR_AZIMUTH"),
            "orbit": img.get("ORBIT"),
        })

    feats = col.map(describe).getInfo()["features"]
    scenes = [f["properties"] for f in feats]
    return sorted(scenes, key=lambda s: s["datetime"])


def download_emit_scene(scene_id: str, grid: Grid, out_dir: Path, max_request_mb: float, retries: int) -> Path:
    """Reflectance, uncertainty and QA for one scene on ``grid`` (cached)."""
    ee = _ee()
    scene_dir = Path(out_dir) / scene_id
    done = scene_dir / "meta.json"
    if done.exists():
        log.info("EMIT %s already downloaded", scene_id)
        return scene_dir

    img = ee.Image(f"{EMIT_L2A}/{scene_id}")
    info = img.toDictionary(["reflectance_wavelengths", "reflectance_fwhm", "SOLAR_ZENITH",
                             "SOLAR_AZIMUTH", "ORBIT", "SCENE"]).getInfo()
    wl = np.asarray(info["reflectance_wavelengths"], dtype=float)
    fwhm = np.asarray(info["reflectance_fwhm"], dtype=float)
    n = len(wl)
    refl_bands = [f"reflectance_{i}" for i in range(n)]
    unc_bands = [f"reflectance_uncertainity_{i}" for i in range(n)]  # sic: EE catalog spelling

    log.info("EMIT %s: downloading %d reflectance bands on %dx%d grid", scene_id, n, grid.width, grid.height)
    refl = fetch(img, refl_bands, grid, max_request_mb, retries)
    unc = fetch(img, unc_bands, grid, max_request_mb, retries)
    qa = fetch(img, EMIT_QA_BANDS, grid, max_request_mb, retries)

    names = [f"{w:.2f}nm" for w in wl]
    write_raster(scene_dir / "reflectance.tif", refl, grid, names)
    write_raster(scene_dir / "uncertainty.tif", unc, grid, names)
    write_raster(scene_dir / "qa.tif", qa, grid, EMIT_QA_BANDS)
    meta = {"scene_id": scene_id, "collection": EMIT_L2A,
            "datetime": ee.Image(img).date().format("YYYY-MM-dd'T'HH:mm:ss").getInfo(),
            "wavelengths_nm": wl.tolist(), "fwhm_nm": fwhm.tolist(),
            "solar_zenith": info.get("SOLAR_ZENITH"), "solar_azimuth": info.get("SOLAR_AZIMUTH"),
            "orbit": info.get("ORBIT"), "scene": info.get("SCENE"), "grid": grid.to_dict(),
            "resampling": "nearest (Earth Engine default) from native 60 m EPSG:4326"}
    done.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return scene_dir


# ---------------------------------------------------------------- SRTM + ASTER GED

def download_srtm(grid: Grid, path: Path, max_request_mb: float, retries: int) -> Path:
    if path.exists():
        return path
    ee = _ee()
    img = ee.Image(SRTM).select("elevation").resample("bilinear")
    dem = fetch(img, ["elevation"], grid, max_request_mb, retries)
    write_raster(path, dem, grid, ["elevation_m"], tags={"source": SRTM, "resampling": "bilinear"})
    return path


DEMS = {"srtm": (SRTM, None, "elevation", "SRTM"),
        "aw3d30": ("JAXA/ALOS/AW3D30/V4_1", "collection", "DSM", "ALOS AW3D30"),
        "copernicus": ("COPERNICUS/DEM/GLO30", "collection", "DEM", "Copernicus GLO-30")}


def dem_label(source: str) -> str:
    return DEMS[source][3]


def dem_path(raw_dir: Path, source: str) -> Path:
    """raw/srtm/dem_30m_padded.tif for SRTM (unchanged), raw/<source>/dem_30m_padded.tif otherwise."""
    return raw_dir / source / "dem_30m_padded.tif"


def download_dem(source: str, grid: Grid, path: Path, max_request_mb: float, retries: int) -> Path:
    if source == "srtm":
        return download_srtm(grid, path, max_request_mb, retries)
    if path.exists():
        return path
    ee = _ee()
    asset, _, band, _ = DEMS[source]
    region = ee.Geometry.Rectangle(list(grid_bounds_wgs84(grid)))
    coll = ee.ImageCollection(asset).filterBounds(region).select(band)
    # tiles share one projection; set it on the mosaic so bilinear resampling uses native pixels
    img = coll.mosaic().setDefaultProjection(coll.first().projection()).resample("bilinear")
    dem = fetch(img, [band], grid, max_request_mb, retries)
    write_raster(path, dem, grid, ["elevation_m"], tags={"source": asset, "resampling": "bilinear"})
    return path


def grid_bounds_wgs84(grid: Grid):
    from pyproj import Transformer
    x0, y0, x1, y1 = grid.bounds
    t = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    xs, ys = t.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return min(xs), min(ys), max(xs), max(ys)


def download_aster_ged(grid: Grid, path: Path, max_request_mb: float, retries: int) -> Path:
    if path.exists():
        return path
    ee = _ee()
    img = ee.Image(ASTER_GED).select(GED_BANDS).resample("bilinear")
    data = fetch(img, GED_BANDS, grid, max_request_mb, retries)
    for i, band in enumerate(GED_BANDS):
        data[i] *= GED_SCALE.get(band, 1.0)
    emis = data[:5]
    if np.nanmedian(emis) < 0.5 or np.nanmedian(emis) > 1.0:
        raise ValueError(f"ASTER GED emissivity median {np.nanmedian(emis):.3f} outside 0.5-1.0; check scale factors")
    write_raster(path, data, grid, GED_BANDS,
                 tags={"source": ASTER_GED, "resampling": "bilinear from native 100 m", "scaled": "true"})
    return path
