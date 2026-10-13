import numpy as np

from geoscan.config import AlterationConfig, TargetConfig
from geoscan.grid import Grid
from geoscan.prospectivity.alteration import normalise, potential_field
from geoscan.prospectivity.targets import cpi, location_name, select_targets


def test_potential_field_peaks_on_detections_and_decays():
    depth = np.zeros((60, 60))
    weight = np.zeros((60, 60))
    depth[28:32, 28:32], weight[28:32, 28:32] = 0.05, 1.0
    f = potential_field(depth, weight, 60.0, AlterationConfig())
    assert f.argmax() == np.ravel_multi_index((29, 29), f.shape) or f[29:31, 29:31].max() == f.max()
    assert f[30, 30] > 10 * f[30, 5]           # far from the cluster the field has decayed
    assert f.max() <= 0.05 + 1e-6              # never exceeds the observed depth


def test_normalise_uses_p98():
    field = np.linspace(0, 1, 10001).reshape(1, -1).astype("float32")
    a, ref = normalise(field, np.ones_like(field, bool), 98.0)
    assert abs(ref - 0.98) < 1e-3 and a.max() == 1.0


def test_cpi_formula_and_target_spacing():
    g = Grid("EPSG:32644", 0.0, 3000.0, 30.0, 100, 100)
    ones = np.ones(g.shape, "float32")
    a = {"muscovite": ones, "quartz": ones, "gossan": ones, "chlorite": ones}
    c, h = cpi(a, ones, np.ones(g.shape, bool), TargetConfig())
    assert np.allclose(c, 1.0) and np.allclose(h, 1.0)

    rng = np.random.default_rng(0)
    c = rng.uniform(0.3, 0.9, g.shape).astype("float32")
    df = select_targets(c, g, {}, np.ones(g.shape, bool), TargetConfig())
    assert len(df) == 50
    xy = df[["Easting", "Northing"]].to_numpy()
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    assert d[np.triu_indices(len(df), 1)].min() >= 250.0
    assert df["CPI"].is_monotonic_decreasing
    assert location_name(1) == "Location A" and location_name(27) == "Location AA"
