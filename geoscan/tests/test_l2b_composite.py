import numpy as np

from geoscan.config import SpectralConfig
from geoscan.spectral.composite import composite
from geoscan.spectral.features import FEATURES
from geoscan.spectral.l2b import classify_name
from geoscan.spectral.scene import QA_VALID, QA_VEGETATION, SceneResult


def test_classify_uses_primary_constituent():
    assert classify_name("Calcite+.33Muscov AMX5 Ruby W1R1Ba") == "carbonate"
    assert classify_name("Muscovite+Chlorite CU91-253D W2R4Na") == "white_mica"
    assert classify_name("Goeth+qtz.5+Jarosite.5 AMX11 W1R1BbS") == "ferric_oxide"
    assert classify_name("Prehnite+.67Chlorite AMIX") == "prehnite"
    assert classify_name("Nanohematite BR93-34B2 W1R1BbS") == "ferric_oxide"
    assert classify_name("Plastic_Tarp GDS339 Green W1R1Fa") is None


def _scene(depth, sigma, position, qa):
    layers = {"qa": np.array(qa, dtype="float32")}
    for f in FEATURES:
        layers[f"{f.key}_depth"] = np.array(depth, dtype="float32")
        layers[f"{f.key}_depth_sigma"] = np.array(sigma, dtype="float32")
        layers[f"{f.key}_position"] = np.array(position, dtype="float32")
        layers[f"{f.key}_position_sigma"] = np.full(len(depth), 1.0, dtype="float32")
    return SceneResult("s", "2024-01-01", layers, {})


def test_composite_weighting_masking_and_edge_rule():
    cfg = SpectralConfig(detect_sigma=2.0, min_depth=0.01)
    a = _scene([0.05, 0.05, 0.05], [0.01, 0.01, 0.01], [2205, np.nan, 2205], [QA_VALID, QA_VALID, QA_VEGETATION])
    b = _scene([0.03, 0.05, 0.05], [0.02, 0.01, 0.01], [2207, np.nan, 2205], [QA_VALID, QA_VALID, QA_VEGETATION])
    out = composite([a, b], cfg)["aloh_2200"]
    # inverse-variance mean of 0.05 (w=10000) and 0.03 (w=2500)
    assert np.isclose(out["depth"][0], (0.05 * 10000 + 0.03 * 2500) / 12500, atol=1e-6)
    assert out["detected"][0] == 1
    # deep but minimum always on the window edge -> not detected
    assert out["detected"][1] == 0
    # vegetation-masked in every scene -> no observation
    assert np.isnan(out["n_valid"][2])
