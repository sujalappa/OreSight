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

## Data sources and credits

OreSight is built on data from these organisations. The data remain the property of their providers and are used under their terms.

| Data | Used for | Provider and credit |
|---|---|---|
| EMIT L2A surface reflectance and L2B mineral identification | Muscovite, chlorite and gossan layers | NASA EMIT mission, NASA Jet Propulsion Laboratory; distributed by NASA LP DAAC. [doi:10.5067/EMIT/EMITL2ARFL.001](https://doi.org/10.5067/EMIT/EMITL2ARFL.001), [doi:10.5067/EMIT/EMITL2BMIN.001](https://doi.org/10.5067/EMIT/EMITL2BMIN.001) |
| ASTER Global Emissivity Dataset (AG100 v003) | Quartz layer | NASA Jet Propulsion Laboratory; distributed by NASA LP DAAC. [doi:10.5067/Community/ASTER_GED/AG100.003](https://doi.org/10.5067/Community/ASTER_GED/AG100.003) |
| ALOS World 3D 30 m (AW3D30) | Lineaments (default) | © Japan Aerospace Exploration Agency (JAXA) |
| SRTM 1 arc-second (SRTMGL1 v003) | Lineaments (optional) | NASA / USGS; distributed by NASA LP DAAC. [doi:10.5067/MEaSUREs/SRTM/SRTMGL1.003](https://doi.org/10.5067/MEaSUREs/SRTM/SRTMGL1.003) |
| Copernicus DEM GLO-30 | Lineaments (optional) | © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA; all rights reserved. [doi:10.5270/ESA-c5d3d65](https://doi.org/10.5270/ESA-c5d3d65) |
| Ground gravity and magnetic stations, stream sediment geochemistry, geomorphology, mapped lineaments | Reference layers and checks | Geological Survey of India (GSI), Ministry of Mines, Government of India; Bhukosh portal |
| Exploration report index, geological maps, lode and borehole data | Validation | National Geoscience Data Repository (NGDR), Geological Survey of India, and Mineral Exploration and Consultancy Limited (MECL) |
| Satellite data access | Fetching imagery for an area | Google Earth Engine |
| Base maps | Map background | Esri World Imagery; OpenTopoMap (CC-BY-SA), © OpenStreetMap contributors |

No data is included in this repository. Exploration data used for validation is not public.

## Limits

- The method sees the surface. It cannot assess ground hidden by younger cover rocks or dense vegetation, and it can miss structures that have no surface expression.
- Mineral assignments are spectral: an Al-OH absorption read as white mica can also come from kaolinite or smectite, and iron oxide from satellites cannot separate a gossan from laterite.
- The CPI weights are expert-assigned and follow one deposit model. Other deposit types may need different weights.
- Validation so far rests on a small number of areas with known mineralisation.
