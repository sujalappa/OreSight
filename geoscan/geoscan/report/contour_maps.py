"""Publication-style contour maps in longitude / latitude.

Filled contours of a smooth surface, labelled isolines, clipped to the AOI, over an SRTM
hillshade, with lineaments, scale bar and north arrow. Areas with no usable data are shaded
and labelled rather than filled.

Surfaces are resampled finer (bicubic) for drawing only; the GeoTIFFs keep native resolution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy.ndimage import distance_transform_edt, zoom  # noqa: E402

from ..grid import Grid  # noqa: E402
from .maps import save_figure  # noqa: E402

STROKE = [pe.withStroke(linewidth=2.2, foreground="white")]
matplotlib.rcParams["hatch.color"] = "#a8a8a8"
matplotlib.rcParams["hatch.linewidth"] = 0.5


def lonlat_mesh(grid: Grid, factor: int = 1):
    """Cell-centre lon/lat arrays for ``grid`` subdivided by ``factor``."""
    res = grid.res / factor
    xs = grid.x0 + (np.arange(grid.width * factor) + 0.5) * res
    ys = grid.y0 - (np.arange(grid.height * factor) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(X, Y)
    return lon, lat


def _fill_nearest(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if valid.all() or not valid.any():
        return np.where(valid, arr, 0.0)
    _, (ri, ci) = distance_transform_edt(~valid, return_indices=True)
    return arr[ri, ci]


def upsample(arr: np.ndarray, factor: int, support: np.ndarray | None = None,
             min_support: float | None = None) -> np.ndarray:
    """Bicubic resampling for display. With ``support``, the no-data boundary is the
    ``min_support`` isoline of the resampled support surface, so it is smooth too."""
    valid = np.isfinite(arr)
    if factor <= 1:
        return arr
    fine = zoom(_fill_nearest(arr), factor, order=3, grid_mode=True, mode="nearest")
    if support is not None and min_support is not None:
        fine_s = zoom(np.nan_to_num(support, nan=0.0), factor, order=3, grid_mode=True, mode="nearest")
        inside = zoom(np.isfinite(support).astype(float), factor, order=1, grid_mode=True, mode="nearest") > 0.5
        keep = (fine_s >= min_support) & inside
    else:
        keep = zoom(valid.astype(float), factor, order=1, grid_mode=True, mode="nearest") > 0.5
    return np.where(keep, fine, np.nan)


def _aoi_patch(aoi_wgs84, ax):
    verts, codes = [], []
    for poly in getattr(aoi_wgs84, "geoms", [aoi_wgs84]):
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)[:, :2]
            verts.extend(xy)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(verts, codes), transform=ax.transData, facecolor="none", edgecolor="none")


class ContourMap:
    def __init__(self, aoi_wgs84, hillshade=None, lines_wgs84=None, figsize=(10, 8.6)):
        self.aoi = aoi_wgs84
        self.hillshade = hillshade          # (array, grid)
        self.lines = lines_wgs84
        self.figsize = figsize

    def _frame(self, title: str, subtitle: str | None):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=250)
        fig.patch.set_facecolor("white")
        if self.hillshade is not None:
            hs, g = self.hillshade
            lon, lat = lonlat_mesh(g)
            ax.pcolormesh(lon, lat, hs, cmap="gray", shading="gouraud", alpha=0.55, zorder=1, rasterized=True)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=22 if subtitle else 10)
        if subtitle:
            ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                    color="#444444")
        return fig, ax

    def _finish(self, fig, ax, path: Path, legend_handles=None, lines_label=True):
        minx, miny, maxx, maxy = self.aoi.bounds
        for poly in getattr(self.aoi, "geoms", [self.aoi]):
            x, y = poly.exterior.xy
            ax.plot(x, y, color="#111111", lw=1.6, zorder=8)
        handles = list(legend_handles or [])
        if self.lines is not None and len(self.lines):
            for g in self.lines.geometry:
                x, y = g.xy
                ax.plot(x, y, color="#00b8d4", lw=1.1, ls="--", alpha=0.9, zorder=7)
            if lines_label:
                handles.append(Line2D([0], [0], color="#00b8d4", lw=1.3, ls="--",
                                      label=f"Topographic lineaments (n={len(self.lines)})"))
        pad_x, pad_y = 0.02 * (maxx - minx), 0.02 * (maxy - miny)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect(1.0 / np.cos(np.radians((miny + maxy) / 2)))
        ax.set_xlabel("Longitude (°E)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Latitude (°N)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25, ls=":")
        ax.ticklabel_format(useOffset=False, style="plain")

        # 1 km scale bar and north arrow
        km_deg = 1.0 / (111.32 * np.cos(np.radians((miny + maxy) / 2)))
        sx, sy = minx + 3 * pad_x, miny + 2.5 * pad_y
        ax.plot([sx, sx + km_deg], [sy, sy], color="black", lw=3.5, solid_capstyle="butt", zorder=10)
        ax.text(sx + km_deg / 2, sy + 1.2 * pad_y, "1 km", ha="center", fontsize=8.5, fontweight="bold",
                path_effects=STROKE, zorder=10)
        ax.annotate("N", xy=(1.035, 0.99), xytext=(1.035, 0.88), xycoords="axes fraction", ha="center", va="top",
                    fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=2),
                    annotation_clip=False, zorder=10)
        if handles:
            ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=8,
                      framealpha=0.95, edgecolor="#cccccc")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, path, bbox_inches="tight")
        plt.close(fig)

    def surface(self, path: Path, arr: np.ndarray, grid: Grid, title: str, cbar_label: str, *,
                cmap="YlOrRd", vmin=0.0, vmax=100.0, levels=(30, 55, 75), fmt="%d%%", subtitle=None,
                upsample_factor=4, support=None, min_support=None, contour_colors=None, alpha=0.82,
                targets=None, n_labels=0, n_stars=10, hatch_mask=None, hatch_label=None, zones=None, n_zones=12,
                nodata_label="Not assessed (no usable EMIT data)"):
        fig, ax = self._frame(title, subtitle)
        fine = upsample(arr, upsample_factor, support, min_support)
        lon, lat = lonlat_mesh(grid, upsample_factor)
        clip = _aoi_patch(self.aoi, ax)
        ax.add_patch(clip)

        handles = []
        nodata = ~np.isfinite(fine)
        # Only mark gaps that fall inside the AOI (outside it is clipped anyway).
        step = max(1, lon.size // 40000)
        sample = np.column_stack([lon.ravel()[::step], lat.ravel()[::step]])
        in_aoi = clip.get_path().contains_points(sample)
        gap_share = (nodata.ravel()[::step] & in_aoi).sum() / max(in_aoi.sum(), 1)
        gaps_inside = gap_share > 0.01  # ignore slivers along the boundary
        if gaps_inside:
            nd = ax.contourf(lon, lat, nodata.astype(float), levels=[0.5, 1.5], colors=["#e6e6e6"],
                             hatches=["///"], alpha=0.6, zorder=2)
            nd.set_clip_path(clip)
            handles.append(Patch(facecolor="#e6e6e6", hatch="///", edgecolor="#a8a8a8", label=nodata_label))

        fill_levels = np.linspace(vmin, vmax, 41)
        cf = ax.contourf(lon, lat, np.clip(fine, vmin, vmax), levels=fill_levels, cmap=cmap, alpha=alpha,
                         extend="neither", zorder=3)
        cf.set_clip_path(clip)
        lv = [l for l in levels if np.nanmin(fine) < l < np.nanmax(fine)] if np.isfinite(fine).any() else []
        if lv:
            colors = [contour_colors[levels.index(l)] for l in lv] if contour_colors else "#1a1a1a"
            cs = ax.contour(lon, lat, fine, levels=lv, colors=colors, linewidths=np.linspace(0.9, 1.5, len(lv)),
                            alpha=0.9, zorder=5)
            cs.set_clip_path(clip)
            labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt=fmt)
            stroke = [pe.withStroke(linewidth=2.2, foreground="black")] if contour_colors else STROKE
            plt.setp(labels, path_effects=stroke)
        if hatch_mask is not None and np.any(hatch_mask):
            fine_h = zoom(np.asarray(hatch_mask, float), upsample_factor, order=1, grid_mode=True, mode="nearest") \
                if upsample_factor > 1 else np.asarray(hatch_mask, float)
            hh = ax.contourf(lon, lat, fine_h, levels=[0.5, 1.5], colors=["none"], hatches=["\\\\\\\\"], zorder=4)
            hh.set_clip_path(clip)
            handles.append(Patch(facecolor="none", hatch="\\\\\\\\", edgecolor="#555555", label=hatch_label))
        if zones is not None and len(zones):          # zone numbers at each zone's peak (the zones are the result)
            for _, z in zones.head(n_zones).iterrows():
                ax.annotate(z["Zone_ID"], xy=(z["Peak_Longitude"], z["Peak_Latitude"]), xytext=(0, 9),
                            textcoords="offset points", ha="center", va="center",
                            fontsize=5.8, fontweight="bold", color="black", zorder=10,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="black", lw=0.7, alpha=0.92))
            handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor="black",
                                  markersize=8, label=f"Prospective zone = inside the {levels[1]:g} % line "
                                                      f"(top {min(n_zones, len(zones))} numbered)"))
            n_labels = 0                              # points stay as plain markers, without name boxes
        if targets is not None and len(targets):
            top = targets.head(max(n_stars, n_labels))
            if n_labels:
                ax.scatter(top["Longitude"], top["Latitude"], s=70, marker="*", c="white", edgecolors="#d50000",
                           linewidths=1.4, zorder=9)
                span = self.aoi.bounds[2] - self.aoi.bounds[0]
                minx, miny, maxx, maxy = self.aoi.bounds
                r = 0.055 * span
                dirs = [(np.cos(a), np.sin(a)) for a in np.radians([45, 135, 315, 225, 90, 270, 0, 180])]
                placed = [(x, y) for x, y in zip(top["Longitude"], top["Latitude"])]  # avoid the stars too
                for _, t in top.head(n_labels).iterrows():
                    best, best_d = None, -1.0
                    for dx, dy in dirs:
                        for k in (1.0, 1.6):
                            cx, cy = t["Longitude"] + k * r * dx, t["Latitude"] + k * r * dy * 0.8
                            if not (minx <= cx <= maxx and miny <= cy <= maxy):
                                continue
                            d = min(np.hypot((cx - px) / 1.8, cy - py) for px, py in placed)
                            if d > best_d:
                                best, best_d = (cx, cy), d
                    best = best or (t["Longitude"] + r, t["Latitude"] + r)
                    placed.append(best)
                    ax.annotate(f"{t.get('Location_Name', t.get('Target_Name', ''))}\n({t['CPI_Percent']:.0f}%)", xy=(t["Longitude"], t["Latitude"]),
                                xytext=best, ha="center", va="center",
                                fontsize=6.8, fontweight="bold", color="white", zorder=11,
                                bbox=dict(boxstyle="round,pad=0.2", fc="#d50000", ec="white", lw=0.8, alpha=0.95),
                                arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
                handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
                                      markeredgecolor="#d50000", markersize=11, label=f"Top {n_labels} field-check locations"))
            else:
                ax.scatter(top["Longitude"], top["Latitude"], s=55, marker="*", c="#ffd600", edgecolors="black",
                           linewidths=0.8, zorder=9)
                handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffd600",
                                      markeredgecolor="black", markersize=10,
                                      label=f"Suggested field-check locations (top {len(top)})"))
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.07, shrink=0.7,
                          ticks=np.linspace(vmin, vmax, 6))
        cb.set_label(cbar_label, fontsize=9.5, fontweight="bold")
        self._finish(fig, ax, path, handles)

    def lineaments(self, path: Path, lines_wgs84, title: str, subtitle=None):
        fig, ax = self._frame(title, subtitle)
        colors = {"N-S": "#d50000", "NE-SW": "#ff6d00", "E-W": "#2962ff", "NW-SE": "#00c853"}
        lw = 1.0 + 2.5 * (lines_wgs84["length_m"] - lines_wgs84["length_m"].min()) / \
            max(lines_wgs84["length_m"].max() - lines_wgs84["length_m"].min(), 1.0)
        is_scarp = lines_wgs84.get("morphology", "").eq("scarp") if "morphology" in lines_wgs84 else \
            np.zeros(len(lines_wgs84), bool)
        for (_, row), w, scarp in zip(lines_wgs84.iterrows(), lw, is_scarp):
            x, y = row.geometry.xy
            if scarp:
                ax.plot(x, y, color="#757575", lw=max(1.0, 0.7 * w), ls="--", zorder=5)
            else:
                ax.plot(x, y, color=colors[row["strike_class"]], lw=w, solid_capstyle="round", zorder=6)
        structural = lines_wgs84[~np.asarray(is_scarp)]
        counts = structural["strike_class"].value_counts()
        handles = [Line2D([0], [0], color=c, lw=2.5, label=f"{k} (n={counts.get(k, 0)})") for k, c in colors.items()]
        if np.any(is_scarp):
            handles.append(Line2D([0], [0], color="#757575", lw=1.5, ls="--",
                                  label=f"Scarp / escarpment edge (n={int(np.sum(is_scarp))}, not used in CPI)"))
        saved, self.lines = self.lines, None
        self._finish(fig, ax, path, handles)
        self.lines = saved
