import zipfile

import numpy as np
import pytest

from geoscan.aoi import load_aoi
from geoscan.grid import Grid, grid_for_aoi
from geoscan.raster_io import read_raster, write_raster

KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Block</name>
<Polygon><outerBoundaryIs><LinearRing><coordinates>
78.264,21.900,0 78.382,21.894,0 78.383,21.969,0 78.260,21.974,0 78.264,21.900,0
</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>"""


def test_kml_and_kmz(tmp_path):
    kml = tmp_path / "block.kml"
    kml.write_text(KML, encoding="utf-8")
    kmz = tmp_path / "block.kmz"
    with zipfile.ZipFile(kmz, "w") as z:
        z.writestr("doc.kml", KML)
    a, b = load_aoi(kml), load_aoi(kmz)
    assert a.geometry.equals(b.geometry)
    assert a.utm_crs.to_epsg() == 32644
    assert 90 < a.area_km2() < 115


def test_grids_are_aligned_and_north_up(tmp_path):
    kml = tmp_path / "block.kml"
    kml.write_text(KML, encoding="utf-8")
    g30 = grid_for_aoi(load_aoi(kml), 30.0, 180.0)
    g60, g90 = g30.coarsen(2), g30.coarsen(3)
    assert (g30.x0, g30.y0) == (g60.x0, g60.y0) == (g90.x0, g90.y0)
    assert g30.transform.e < 0
    padded = g30.pad(1500)
    rows, cols = g30.window_in(padded)
    assert rows.stop - rows.start == g30.height and cols.stop - cols.start == g30.width


def test_raster_roundtrip_keeps_orientation(tmp_path):
    g = Grid("EPSG:32644", 0.0, 1000.0, 10.0, 4, 3)
    arr = np.arange(12, dtype="float32").reshape(3, 4)  # row 0 = north
    write_raster(tmp_path / "a.tif", arr, g, ["v"])
    data, g2, names, _ = read_raster(tmp_path / "a.tif")
    assert g2 == g and names == ["v"]
    assert np.array_equal(data[0], arr)
    with pytest.raises(ValueError):
        write_raster(tmp_path / "b.tif", arr.T, g)
