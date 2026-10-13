"""ASTER Global Emissivity Dataset (GED, AG100 v003) lithological indices.

Ninomiya et al. (2005) indices from mean emissivity of TIR bands 10-14:

    QI = e11^2 / (e10 * e12)          quartz index (Si-O reststrahlen near 8.6 um; higher = more quartz)
    SI = e13 / e12                    silica index
    CI = e13 / e14                    carbonate index (higher = more carbonate)
    MI = e12 * e14^3 / e13^4          mafic index (higher = more mafic / less silica)

The composite quartz-silicification index used for the quartz alteration layer follows the
Phase-1 methodology (section 4.5): QSI = norm(0.55 * norm(QI) + 0.45 * norm(SI)), with
norm(X) = clip((X - P5) / (P98 - P5), 0, 1) inside the AOI (see ``quartz_silicification``).

GED is a multi-year (2000-2008) mean at 100 m. Its "sdev" layers describe variability
between the averaged observations, not measurement error; they are propagated here by
Monte Carlo to give a spread for each index, which should be read as a stability measure.
Indices are reported raw (no stretching); any normalisation belongs to display only.
"""

from __future__ import annotations

import numpy as np

from ..raster_io import read_raster


def _indices(e10, e11, e12, e13, e14):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"quartz_index": e11 ** 2 / (e10 * e12),
                "silica_index": e13 / e12,
                "carbonate_index": e13 / e14,
                "mafic_index": e12 * e14 ** 3 / e13 ** 4}


def compute_indices(ged_path, draws: int = 64, seed: int = 42):
    data, grid, names, _ = read_raster(ged_path)
    band = {n: data[i] for i, n in enumerate(names)}
    e = [band[f"emissivity_band{b}"] for b in range(10, 15)]
    s = [band[f"emissivity_sdev_band{b}"] for b in range(10, 15)]

    nominal = _indices(*e)
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in nominal}
    for _ in range(draws):
        pert = [ei + si * rng.standard_normal(ei.shape) for ei, si in zip(e, s)]
        for k, v in _indices(*pert).items():
            acc[k].append(v)

    water = band["land_water_map"] > 0
    layers = {}
    for k, v in nominal.items():
        layers[k] = np.where(water, np.nan, v).astype("float32")
        layers[f"{k}_spread"] = np.where(water, np.nan, np.nanstd(np.stack(acc[k]), axis=0)).astype("float32")
    layers["ged_ndvi"] = band["ndvi"]
    layers["ged_num_obs"] = band["num_obs"]
    return layers, grid


def _norm(x: np.ndarray, inside: np.ndarray, lo_p: float = 5.0, hi_p: float = 98.0) -> np.ndarray:
    lo, hi = np.nanpercentile(x[inside], [lo_p, hi_p])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def quartz_silicification(qi: np.ndarray, si: np.ndarray, inside: np.ndarray,
                          w_qi: float = 0.55, w_si: float = 0.45) -> np.ndarray:
    """Composite quartz-silicification index QSI (0-1) per methodology section 4.5."""
    qsi = w_qi * _norm(qi, inside) + w_si * _norm(si, inside)
    return np.where(inside, _norm(qsi, inside), np.nan).astype("float32")
