# OreSight

Copper prospectivity mapping. OreSight maps the surface alteration that copper-bearing fluids leave behind, traces the structures those fluids may have used, and ranks **prospective zones** for field follow-up.

It predicts regions, not drill points. The results are a surface-evidence ranking to guide mapping, sampling and geophysics.

## What it does

1. **Alteration minerals** from NASA EMIT imaging-spectrometer reflectance (285 bands): muscovite / sericite, chlorite / epidote and gossan (iron oxide), using continuum-removed band depths with per-pixel uncertainty. Quartz comes from ASTER thermal emissivity.
2. **Lineaments** traced from a 30 m elevation model (ALOS AW3D30, SRTM or Copernicus GLO-30).
3. **Copper Prospectivity Index (CPI)**: a weighted combination of the four alteration layers and proximity to lineaments.
4. **Prospective zones**: connected ground above a CPI threshold, ranked by tier and peak score, each with its area, main evidence and suggested follow-up. Suggested field-check locations are listed as starting points inside the zones.
5. **Validation** against ground truth, where it exists.

## Web app

Upload an area (KML, KMZ or GeoJSON). The app fetches the satellite data from Google Earth Engine, runs the analysis and shows the results:

- Interactive map with every layer, contour lines, zones, lineaments and reference geophysics / geochemistry
- Publication-style contour maps
- Zones and field-check locations as tables
- Adjustable parameters, each with an explanation of what it is, what changing it does and why the default was chosen. Only the affected steps are recomputed.
- Downloads: maps (PNG), rasters (GeoTIFF), vectors (Shapefile, GeoJSON, GeoPackage) and tables (CSV)
- Optional user accounts with an activity log

```bash
cd geoscan
pip install -r requirements.txt fastapi uvicorn python-multipart
python -m geoscan.webapp
```

The app opens in the browser at `http://127.0.0.1:8765`.

## Command line

```bash
cd geoscan
python -m geoscan run --aoi path/to/area.kml --name my_area --ee-project YOUR_EARTH_ENGINE_PROJECT
```

Raw downloads are cached in `data/<area>/` and results are written to `outputs/<area>/`, with a `manifest.json` describing every file and the settings used.

## Requirements

- Python 3.11 or newer
- A Google Earth Engine project. Sign in once with `earthengine authenticate`, or set `GEE_SERVICE_ACCOUNT_KEY` to a service-account key on a server.
- Set your project id in `geoscan/config/default.yaml` (`acquire.ee_project`) or pass `--ee-project`.

## Repository layout

| Path | Contents |
|---|---|
| `geoscan/geoscan/` | The Python package: data acquisition, spectral features, lineaments, prospectivity, map rendering and the web app (`webapp/`) |
| `geoscan/config/` | Default settings |
| `geoscan/scripts/` | Validation and experiment scripts |
| `geoscan/tests/` | Unit tests (`pytest`) |
| `deploy/huggingface/` | Deployment to a Hugging Face Space |

## Data sources

- NASA EMIT L2A surface reflectance and L2B mineral identification
- ASTER Global Emissivity Dataset
- ALOS AW3D30, SRTM and Copernicus GLO-30 elevation models
- Optional reference layers from the Geological Survey of India (Bhukosh): ground gravity and magnetic stations, stream-sediment geochemistry

No data is included in this repository. Exploration data used for validation is not public.

## Limits

- The method sees the surface. It cannot assess ground hidden by younger cover rocks or dense vegetation, and it can miss structures that have no surface expression.
- Mineral assignments are spectral: an Al-OH absorption read as white mica can also come from kaolinite or smectite, and iron oxide from satellites cannot separate a gossan from laterite.
- The CPI weights are expert-assigned and follow one deposit model. Other deposit types may need different weights.
- Validation so far rests on a small number of areas with known mineralisation.
