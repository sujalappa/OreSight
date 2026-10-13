"""Quick-look PNG maps.

Everything is drawn in grid (UTM) coordinates using ``extent`` = grid bounds and
``origin='upper'``, which is correct for geoscan's north-up arrays, so layers on
different grids (30 / 60 / 90 m) overlay correctly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..grid import Grid  # noqa: E402


def _extent_km(grid: Grid):
    w, s, e, n = grid.bounds
    return [w / 1000, e / 1000, s / 1000, n / 1000]


def _decorate(ax, aoi_geom, lines=None):
    for poly in getattr(aoi_geom, "geoms", [aoi_geom]):
        x, y = poly.exterior.xy
        ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000, color="black", lw=1.2, zorder=6)
    if lines is not None and len(lines):
        for g in lines.geometry:
            x, y = g.xy
            ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000, color="#00bcd4", lw=0.8, alpha=0.9, zorder=5)
    minx, miny, maxx, maxy = aoi_geom.bounds
    pad = 0.03 * (maxx - minx)
    ax.set_xlim((minx - pad) / 1000, (maxx + pad) / 1000)
    ax.set_ylim((miny - pad) / 1000, (maxy + pad) / 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.grid(alpha=0.2, ls=":")
    # scale bar (1 km) and north arrow
    x0, y0 = (minx + pad) / 1000, (miny + pad) / 1000
    ax.plot([x0, x0 + 1], [y0, y0], color="black", lw=3, zorder=7)
    ax.text(x0 + 0.5, y0 + 0.15, "1 km", ha="center", fontsize=8, zorder=7)
    ax.annotate("N", xy=((maxx - pad) / 1000, (maxy - pad) / 1000),
                xytext=((maxx - pad) / 1000, (maxy - pad) / 1000 - 1.0),
                ha="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black"), zorder=7)


def layer_map(path: Path, arr: np.ndarray, grid: Grid, title: str, cbar_label: str, aoi_geom, *,
              cmap="viridis", vmin=None, vmax=None, background=None, lines=None, subtitle: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    if background is not None:
        bg, bg_grid = background
        ax.imshow(bg, extent=_extent_km(bg_grid), origin="upper", cmap="gray", zorder=1)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        vmin = np.percentile(finite, 2) if vmin is None else vmin
        vmax = np.percentile(finite, 98) if vmax is None else vmax
    im = ax.imshow(np.ma.masked_invalid(arr), extent=_extent_km(grid), origin="upper", cmap=cmap,
                   vmin=vmin, vmax=vmax, alpha=0.85 if background is not None else 1.0,
                   interpolation="nearest", zorder=2)
    _decorate(ax, aoi_geom, lines)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label(cbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path)
    plt.close(fig)


def class_map(path: Path, classes: np.ndarray, grid: Grid, labels: dict[int, str], colors: dict[int, str],
              title: str, aoi_geom, background=None, subtitle: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    if background is not None:
        bg, bg_grid = background
        ax.imshow(bg, extent=_extent_km(bg_grid), origin="upper", cmap="gray", zorder=1)
    present = [c for c in sorted(labels) if np.any(classes == c)]
    lut = np.full(max(labels) + 2, np.nan)
    for i, c in enumerate(present):
        lut[c] = i
    idx = np.where(np.isfinite(classes), lut[np.nan_to_num(classes, nan=-1).astype(int)], np.nan)
    ax.imshow(np.ma.masked_invalid(idx), extent=_extent_km(grid), origin="upper",
              cmap=ListedColormap([colors[c] for c in present] or ["#ffffff"]),
              vmin=-0.5, vmax=max(len(present) - 0.5, 0.5), interpolation="nearest", zorder=2, alpha=0.9)
    _decorate(ax, aoi_geom)
    ax.legend(handles=[Patch(color=colors[c], label=labels[c]) for c in present], loc="upper left",
              bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path, bbox_inches="tight")
    plt.close(fig)


def rose_diagram(path: Path, rose, title: str):
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    width = np.radians(rose["bin_end_deg"] - rose["bin_start_deg"])
    for offset in (0, 180):  # strikes are axial: mirror into the opposite half
        ax.bar(np.radians(rose["bin_start_deg"] + offset), rose["length_km"], width=width, align="edge",
               color="#b71c1c", edgecolor="white", lw=0.5)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def scene_chart(path: Path, scenes_df):
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=200)
    x = np.arange(len(scenes_df))
    ax.bar(x, 100 * scenes_df["aoi_coverage"], color="#cfd8dc", label="AOI covered by scene")
    ax.bar(x, 100 * scenes_df["valid_fraction"], color="#2e7d32", label="Valid pixels (usable)")
    ax.set_xticks(x, [s[:10] for s in scenes_df["datetime"]])
    ax.set_ylabel("% of AOI")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title("EMIT L2A scene usability", fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def save_figure(fig, path: Path, **kw) -> Path:
    """Save a figure; if the target is locked (open in an image viewer on Windows), save
    alongside as ``<name>_new.png`` and warn instead of failing the run."""
    import logging
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, **kw)
        return path
    except OSError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        fig.savefig(alt, **kw)
        logging.getLogger(__name__).warning("%s is locked (open in a viewer?) - saved %s instead", path.name, alt.name)
        return alt
