import numpy as np

from geoscan.spectral.band_depth import FeatureMeasure
from geoscan.spectral.features import FEATURE_BY_KEY

WL = np.linspace(381.0, 2493.0, 285)  # EMIT-like 7.4 nm sampling


def synthetic(depth, centre, width=18.0, n=1):
    base = 0.30 + 0.00002 * (WL - 2000.0)  # sloping continuum
    spec = base * (1.0 - depth * np.exp(-0.5 * ((WL - centre) / width) ** 2))
    return np.tile(spec, (n, 1))


def test_recovers_depth_and_position():
    m = FeatureMeasure(FEATURE_BY_KEY["aloh_2200"], WL, 10.0)
    refl = synthetic(0.08, 2207.0)
    out = m.measure(refl, np.zeros_like(refl), draws=8, rng=np.random.default_rng(0))
    assert abs(out["depth"][0] - 0.08) < 0.01
    assert abs(out["position"][0] - 2207.0) < 2.0
    assert out["depth_sigma"][0] < 1e-6


def test_flat_spectrum_has_no_depth():
    m = FeatureMeasure(FEATURE_BY_KEY["aloh_2200"], WL, 10.0)
    refl = synthetic(0.0, 2207.0)
    out = m.measure(refl, np.zeros_like(refl), draws=8, rng=np.random.default_rng(0))
    assert abs(out["depth"][0]) < 1e-3


def test_uncertainty_scales_with_noise():
    m = FeatureMeasure(FEATURE_BY_KEY["aloh_2200"], WL, 10.0)
    refl = synthetic(0.05, 2205.0, n=200)
    rng = np.random.default_rng(1)
    s_lo = m.measure(refl, np.full_like(refl, 0.001), 64, rng)["depth_sigma"].mean()
    s_hi = m.measure(refl, np.full_like(refl, 0.004), 64, rng)["depth_sigma"].mean()
    assert 3.0 < s_hi / s_lo < 5.5
