"""Continuum-removed band depth and absorption position with Monte-Carlo uncertainty.

For one feature and one spectrum R(lambda):

    R_L, R_R  = mean reflectance of the bands within +/- hw of each shoulder
    C(lambda) = straight line through (lambda_L, R_L) and (lambda_R, R_R)
    CR        = R / C over the search window
    (lambda_min, CR_min) = vertex of a parabola through the 3 bands around argmin(CR)
    depth     = 1 - CR_min            (signed: negative means no absorption)
    position  = lambda_min            (NaN when the minimum sits on the window edge)

Uncertainty: the spectrum is perturbed ``mc_draws`` times with independent Gaussian
noise of the per-band EMIT L2A reflectance uncertainty, and the standard deviation
of depth and position over the draws is reported. Band-to-band error correlation
is not modelled, so sigma is a lower bound where errors are correlated.
"""

from __future__ import annotations

import numpy as np

from .features import AbsorptionFeature


def band_indices(wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    idx = np.where((wl >= lo) & (wl <= hi))[0]
    if idx.size == 0:
        raise ValueError(f"No bands between {lo} and {hi} nm")
    return idx


def window_mean(refl: np.ndarray, wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.nanmean(refl[..., band_indices(wl, lo, hi)], axis=-1)


class FeatureMeasure:
    """Pre-computes band indices for one feature on one wavelength table."""

    def __init__(self, feature: AbsorptionFeature, wl: np.ndarray, shoulder_hw: float):
        self.feature = feature
        self.L = band_indices(wl, feature.left_nm - shoulder_hw, feature.left_nm + shoulder_hw)
        self.R = band_indices(wl, feature.right_nm - shoulder_hw, feature.right_nm + shoulder_hw)
        self.W = band_indices(wl, *feature.window_nm)
        if self.W.size < 3:
            raise ValueError(f"{feature.key}: fewer than 3 bands in search window")
        self.used = np.unique(np.concatenate([self.L, self.W, self.R]))
        pos = {b: i for i, b in enumerate(self.used)}
        self._L = np.array([pos[b] for b in self.L])
        self._R = np.array([pos[b] for b in self.R])
        self._W = np.array([pos[b] for b in self.W])
        self.lam_L = float(wl[self.L].mean())
        self.lam_R = float(wl[self.R].mean())
        self.lam_W = wl[self.W].astype("float64")

    def _measure(self, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """spec: (..., n_used). Returns depth, position with shape spec.shape[:-1]."""
        rl = spec[..., self._L].mean(axis=-1)
        rr = spec[..., self._R].mean(axis=-1)
        slope = (rr - rl) / (self.lam_R - self.lam_L)
        cont = rl[..., None] + (self.lam_W - self.lam_L) * slope[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            cr = spec[..., self._W] / cont

        n = cr.shape[-1]
        finite = np.isfinite(cr)
        i = np.argmin(np.where(finite, cr, np.inf), axis=-1)
        edge = (i == 0) | (i == n - 1)
        ic = np.clip(i, 1, n - 2)
        y0 = np.take_along_axis(cr, (ic - 1)[..., None], -1)[..., 0]
        y1 = np.take_along_axis(cr, ic[..., None], -1)[..., 0]
        y2 = np.take_along_axis(cr, (ic + 1)[..., None], -1)[..., 0]
        x0, x1, x2 = self.lam_W[ic - 1], self.lam_W[ic], self.lam_W[ic + 1]

        # Vertex of the parabola through three (possibly unevenly spaced) points,
        # in Newton form y = y0 + d1 (x - x0) + a (x - x0)(x - x1).
        d1 = (y1 - y0) / (x1 - x0)
        d2 = (y2 - y1) / (x2 - x1)
        a = (d2 - d1) / (x2 - x0)
        with np.errstate(divide="ignore", invalid="ignore"):
            xv = np.where(a > 0, 0.5 * (x0 + x1) - d1 / (2 * a), x1)
        xv = np.clip(xv, x0, x2)
        yv = np.where(a > 0, y0 + d1 * (xv - x0) + a * (xv - x0) * (xv - x1), y1)

        cr_min = cr.min(axis=-1, initial=np.inf, where=finite)
        depth = 1.0 - np.where(edge, cr_min, yv)
        depth = np.where(np.isfinite(depth) & finite.any(axis=-1), depth, np.nan)
        position = np.where(edge | ~np.isfinite(depth), np.nan, xv)
        return depth, position

    def measure(self, refl: np.ndarray, unc: np.ndarray, draws: int, rng: np.random.Generator,
                chunk: int = 4096) -> dict[str, np.ndarray]:
        """refl, unc: (n_pixels, n_bands). Returns depth, depth_sigma, position, position_sigma."""
        n_px = refl.shape[0]
        out = {k: np.full(n_px, np.nan, dtype="float32")
               for k in ("depth", "depth_sigma", "position", "position_sigma")}
        r_used = refl[:, self.used].astype("float64")
        u_used = unc[:, self.used].astype("float64")
        for s in range(0, n_px, chunk):
            r = r_used[s:s + chunk]
            u = np.nan_to_num(u_used[s:s + chunk], nan=0.0)
            depth, pos = self._measure(r)
            noise = rng.standard_normal((r.shape[0], draws, r.shape[1]))
            d_mc, p_mc = self._measure(r[:, None, :] + noise * u[:, None, :])
            out["depth"][s:s + chunk] = depth
            out["position"][s:s + chunk] = pos
            out["depth_sigma"][s:s + chunk] = np.nanstd(d_mc, axis=1)
            with np.errstate(invalid="ignore"):
                ok = np.isfinite(p_mc).sum(axis=1) >= draws // 2
                out["position_sigma"][s:s + chunk] = np.where(ok, np.nanstd(p_mc, axis=1), np.nan)
        return out
