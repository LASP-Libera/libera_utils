#!/usr/bin/env python
"""Verify footprint matching readers against real ancillary data files.

Loads one 2° × 2° tile from each supplied reader, prints data statistics to
stdout, and writes PNG visualizations to an output directory.

Usage
-----
Run from the repo root (or any directory on the Python path)::

    python -m libera_utils.footprint_matching.verify_readers \\
        --igbp   /data/MCD12Q1.A2023001.h09v05.061.hdf \\
        --nise   /data/NISE_SSMISF18_20260115.HDFEOS \\
        --era5   /data/era5_wind_20230101.nc \\
        --viirs  /data/CLDPROP_D3_VIIRS_NOAA20.A2026147.011.nc \\
        --brdf   /data/VJ143C1.A2026153.002.h5 \\
        --aod    /data/AERDB_D3_GEOLEO_Merged.A2020121.001.nc \\
        --ssf    /data/CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115.nc \\
        --cldpix /data/CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc \\
        --lat-center 45.0 --lon-center -90.0 \\
        --output-dir ./reader_verification

Any subset of reader arguments may be supplied; omitted readers are skipped.

Arguments
---------
--igbp PATH      MCD12Q1 HDF4 tile (download: https://appeears.earthdatacloud.nasa.gov/)
--nise PATH      NISE HDF-EOS4 file (download: https://n5eil01u.ecs.nsidc.org/NISE/)
--era5 PATH      ERA5 NetCDF4 with u10/v10 (download: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
--viirs PATH     CLDPROP_D3 VIIRS NetCDF4 (download: https://www.ncei.noaa.gov/data/cloud-properties-viirs/access/)
--brdf PATH      VJ143C1 VIIRS BRDF HDF5 (download: https://e4ftl01.cr.usgs.gov/VIIRS/VJ143C1.002/)
--aod PATH       AERDB_D3_GEOLEO merged AOD NetCDF4 (Deep Blue GEO-LEO merged daily L3)
--ssf PATH       CERES SSF or FLASHFlux NetCDF4 (download: https://ceres.larc.nasa.gov/data/)
--cldpix PATH    CERES CLDPIX NetCDF4 (download: https://ceres.larc.nasa.gov/data/)
--lat-center     Latitude of the tile center in degrees (default: 45.0)
--lon-center     Longitude of the tile center in degrees (default: -90.0)
--output-dir     Directory for PNG output (default: current directory)
--no-plots       Print statistics only; skip matplotlib visualization
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path

import numpy as np

# Register all built-in readers via __init_subclass__
import libera_utils.footprint_matching.readers  # noqa: F401
from libera_utils.footprint_matching.readers.base import TILE_SIZE_DEG, GriddedDataReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import GridTile, TileKey

# ---------------------------------------------------------------------------
# IGBP land cover class names (IGBP scheme, MCD12Q1 Type 1)
# Source: https://lpdaac.usgs.gov/documents/1409/MCD12_User_Guide_V61.pdf Table 2
# ---------------------------------------------------------------------------
_IGBP_CLASS_NAMES = {
    0: "Water",
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forest",
    6: "Closed Shrubland",
    7: "Open Shrubland",
    8: "Woody Savanna",
    9: "Savanna",
    10: "Grassland",
    11: "Permanent Wetland",
    12: "Cropland",
    13: "Urban / Built-up",
    14: "Cropland / Natural Veg. Mosaic",
    15: "Permanent Snow and Ice",
    16: "Barren / Sparsely Vegetated",
    17: "Water Body (alt.)",
    18: "Wooded Tundra",
    19: "Mixed Tundra",
}


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------


def latlon_to_tile_key(source: str, lat: float, lon: float) -> TileKey:
    """Convert a geographic point to the TileKey of the 2° tile containing it.

    Parameters
    ----------
    source : str
        Reader registry key (used as TileKey.source).
    lat, lon : float
        Latitude and longitude in degrees.

    Returns
    -------
    TileKey
        The tile that contains (lat, lon).
    """
    lat_idx = int(math.floor((lat + 90.0) / TILE_SIZE_DEG))
    lon_idx = int(math.floor((lon + 180.0) / TILE_SIZE_DEG))
    # Clamp to valid grid range (0–89 for lat, 0–179 for lon)
    lat_idx = max(0, min(lat_idx, 89))
    lon_idx = max(0, min(lon_idx, 179))
    return TileKey(source, lat_idx, lon_idx)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _print_tile_summary(tile: GridTile, reader_key: str) -> None:
    """Print a structured summary of a loaded GridTile to stdout."""
    bbox = tile.bounds
    sep = "─" * 60
    print(sep)
    print(f"  Reader : {reader_key}  (source={tile.source})")
    print(f"  BBox   : lat [{bbox.lat_min:.2f}, {bbox.lat_max:.2f}]  lon [{bbox.lon_min:.2f}, {bbox.lon_max:.2f}]")
    print(f"  Shape  : {tile.data.shape}  dtype={tile.data.dtype}")
    print(f"  Coords : {tile.lats.size} lats × {tile.lons.size} lons")
    if tile.timestamp_source:
        print(f"  Timestamp source: {tile.timestamp_source}")
    print(f"  Memory : {tile.nbytes / 1024:.1f} KiB")

    # Per-variable statistics
    data = tile.data
    if data.ndim == 2:
        _print_var_stats(data, name=ReaderRegistry.get(reader_key).VARIABLES[0].name)
    else:
        # 3D: (n_vars, n_lat, n_lon)
        reader_cls = ReaderRegistry.get(reader_key)
        for i, var_spec in enumerate(reader_cls.VARIABLES):
            _print_var_stats(data[i], name=var_spec.name)
    print()


def _print_var_stats(arr: np.ndarray, name: str) -> None:
    """Print min/max/mean/std or a value histogram for one variable slice."""
    valid = arr[arr != 255]  # exclude IGBP fill (255)
    finite = valid[np.isfinite(valid)]

    print(f"\n  [{name}]")
    if finite.size == 0:
        print("    All pixels are fill or NaN — no valid data in this tile.")
        return

    print(f"    valid pixels : {finite.size:,} / {arr.size:,} ({100 * finite.size / arr.size:.1f}%)")
    print(f"    min / max    : {finite.min():.4g} / {finite.max():.4g}")
    print(f"    mean ± std   : {finite.mean():.4g} ± {finite.std():.4g}")

    # For categorical-looking data (small integer range), show a value histogram
    unique_vals = np.unique(finite.astype(int))
    if unique_vals.size <= 20 and np.all(finite == finite.astype(int)):
        print("    Category counts:")
        total = finite.size
        for val in unique_vals:
            count = int(np.sum(finite == val))
            pct = 100.0 * count / total
            label = _IGBP_CLASS_NAMES.get(int(val)) or ""
            label_str = f"  {label}" if label else ""
            print(f"      {int(val):3d}{label_str:<35s}  {count:6,}  ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _make_igbp_figure(tile: GridTile, out_path: Path) -> None:
    """Save a categorical land cover map for IGBP data."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    data = tile.data.copy()
    # Replace fill (255) with NaN for display
    data[data == 255] = np.nan

    # Build a 20-color discrete colormap matched to IGBP class indices 0–19.
    # Colors chosen to match conventional IGBP land cover palette.
    igbp_colors = [
        "#4169E1",  # 0 Water — royal blue
        "#006400",  # 1 Evergreen Needleleaf Forest — dark green
        "#228B22",  # 2 Evergreen Broadleaf Forest — forest green
        "#8FBC8F",  # 3 Deciduous Needleleaf Forest — dark sea green
        "#90EE90",  # 4 Deciduous Broadleaf Forest — light green
        "#6B8E23",  # 5 Mixed Forest — olive drab
        "#A0522D",  # 6 Closed Shrubland — sienna
        "#D2B48C",  # 7 Open Shrubland — tan
        "#9ACD32",  # 8 Woody Savanna — yellow green
        "#F0E68C",  # 9 Savanna — khaki
        "#ADFF2F",  # 10 Grassland — green yellow
        "#4682B4",  # 11 Permanent Wetland — steel blue
        "#FFD700",  # 12 Cropland — gold
        "#FF4500",  # 13 Urban — orange red
        "#BDB76B",  # 14 Cropland/Nat Veg Mosaic — dark khaki
        "#FFFAFA",  # 15 Snow/Ice — snow
        "#808080",  # 16 Barren — gray
        "#00BFFF",  # 17 Water Body (alt.) — deep sky blue
        "#556B2F",  # 18 Wooded Tundra — dark olive green
        "#8B8682",  # 19 Mixed Tundra — warm gray
    ]
    cmap = ListedColormap(igbp_colors)
    bounds = np.arange(-0.5, 20.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(9, 7))
    bbox = tile.bounds
    ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        extent=[bbox.lon_min, bbox.lon_max, bbox.lat_min, bbox.lat_max],
        origin="lower",
        aspect="auto",
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        f"IGBP Land Cover — MCD12Q1\n"
        f"Tile: lat [{bbox.lat_min:.1f}, {bbox.lat_max:.1f}]  "
        f"lon [{bbox.lon_min:.1f}, {bbox.lon_max:.1f}]  "
        f"({tile.data.shape[1]}×{tile.data.shape[0]} px)"
    )

    # Legend for present classes only
    present_vals = sorted(int(v) for v in np.unique(data[~np.isnan(data)]) if 0 <= int(v) <= 19)
    legend_patches = [Patch(color=igbp_colors[v], label=f"{v}: {_IGBP_CLASS_NAMES.get(v, '?')}") for v in present_vals]
    ax.legend(
        handles=legend_patches,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        title="IGBP Class",
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [igbp] Saved: {out_path}")


def _make_nise_figure(tile: GridTile, out_path: Path) -> None:
    """Save a panel of the five NISE surface-coverage layers.

    The NISE reader now emits a 3-D ``(5, n_lat, n_lon)`` tile — one fractional
    coverage layer per Extent code group — so we render a small multi-panel
    figure (one subplot per layer) rather than a single concentration map.
    """
    import matplotlib.pyplot as plt

    bbox = tile.bounds
    var_names = [v.name for v in ReaderRegistry.get("nise").VARIABLES]
    data = tile.data.astype(float)

    # data is (n_vars, n_lat, n_lon); lay the panels out in a single row.
    n_vars = data.shape[0]
    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4.5), squeeze=False)
    extent = [bbox.lon_min, bbox.lon_max, bbox.lat_min, bbox.lat_max]
    for i, name in enumerate(var_names):
        ax = axes[0][i]
        im = ax.imshow(
            data[i],
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            origin="lower",
            aspect="auto",
        )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Longitude (°)")
        if i == 0:
            ax.set_ylabel("Latitude (°)")
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle(
        f"NISE surface coverage layers  —  "
        f"lat [{bbox.lat_min:.1f}, {bbox.lat_max:.1f}]  "
        f"lon [{bbox.lon_min:.1f}, {bbox.lon_max:.1f}]  "
        f"({data.shape[2]}×{data.shape[1]} px)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [nise] Saved: {out_path}")


def _make_era5_figure(tile: GridTile, out_path: Path) -> None:
    """Save a 4-panel ERA5 wind figure: u10, v10, wind speed, wind direction."""
    import matplotlib.pyplot as plt

    u10 = tile.data[0]  # (n_lat, n_lon)
    v10 = tile.data[1]
    speed = np.sqrt(u10**2 + v10**2)
    direction = np.degrees(np.arctan2(u10, v10)) % 360  # met. convention: from north

    bbox = tile.bounds
    lons = tile.lons
    lats = tile.lats
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"ERA5 10-m Wind\n"
        f"Tile: lat [{bbox.lat_min:.1f}, {bbox.lat_max:.1f}]  "
        f"lon [{bbox.lon_min:.1f}, {bbox.lon_max:.1f}]  "
        f"({u10.shape[1]}×{u10.shape[0]} px)",
        fontsize=12,
    )

    panels = [
        (axes[0, 0], u10, "U10 — Zonal Wind (m s⁻¹)", "RdBu_r", None, None),
        (axes[0, 1], v10, "V10 — Meridional Wind (m s⁻¹)", "RdBu_r", None, None),
        (axes[1, 0], speed, "Wind Speed (m s⁻¹)", "YlOrRd", 0, None),
        (axes[1, 1], direction, "Wind Direction (° from N, met. conv.)", "twilight", 0, 360),
    ]

    for ax, arr, title, cmap, vmin, vmax in panels:
        kw = dict(extent=extent, origin="lower", aspect="auto", cmap=cmap)
        if vmin is not None:
            kw["vmin"] = vmin
        if vmax is not None:
            kw["vmax"] = vmax
        else:
            # For diverging colormaps, center on zero
            if cmap == "RdBu_r":
                absmax = max(abs(float(arr.min())), abs(float(arr.max())), 0.001)
                kw["vmin"] = -absmax
                kw["vmax"] = absmax
        im = ax.imshow(arr, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Longitude (°)", fontsize=7)
        ax.set_ylabel("Latitude (°)", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [era5] Saved: {out_path}")


def _make_viirs_cloud_figure(tile: GridTile, out_path: Path) -> None:
    """Save a 3-panel VIIRS cloud property figure."""
    import matplotlib.pyplot as plt

    cmaps = ["Blues", "YlOrBr", "RdPu_r"]
    labels = [
        "Cloud Fraction (0–1)",
        "Cloud Optical Thickness",
        "Cloud Top Pressure (hPa)",
    ]

    bbox = tile.bounds
    lons = tile.lons
    lats = tile.lats
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"VIIRS Cloud Properties (CLDPROP_D3)\n"
        f"Tile: lat [{bbox.lat_min:.1f}, {bbox.lat_max:.1f}]  "
        f"lon [{bbox.lon_min:.1f}, {bbox.lon_max:.1f}]  "
        f"({tile.data.shape[2]}×{tile.data.shape[1]} px)",
        fontsize=12,
    )

    for i, (ax, label, cmap) in enumerate(zip(axes, labels, cmaps)):
        arr_disp = tile.data[i].copy().astype(float)
        im = ax.imshow(
            arr_disp,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Longitude (°)", fontsize=7)
        ax.set_ylabel("Latitude (°)", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viirs_cloud] Saved: {out_path}")


def _make_brdf_figure(tile: GridTile, out_path: Path) -> None:
    """Save a 3×3 panel figure of VIIRS BRDF kernel parameters."""
    import matplotlib.pyplot as plt

    bands = ["shortwave", "vis", "nir"]
    params = ["fiso", "fvol", "fgeo"]
    param_labels = {
        "fiso": "Isotropic (f_iso)",
        "fvol": "Volume (f_vol)",
        "fgeo": "Geometric (f_geo)",
    }

    bbox = tile.bounds
    lons = tile.lons
    lats = tile.lats
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    fig.suptitle(
        f"VIIRS BRDF Kernel Parameters (VJ143C1)\n"
        f"Tile: lat [{bbox.lat_min:.1f}, {bbox.lat_max:.1f}]  "
        f"lon [{bbox.lon_min:.1f}, {bbox.lon_max:.1f}]",
        fontsize=12,
    )

    from libera_utils.footprint_matching.readers.registry import ReaderRegistry

    var_names = [v.name for v in ReaderRegistry.get("viirs_brdf").VARIABLES]

    for row_idx, param in enumerate(params):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            var_name = f"brdf_{band}_{param}"
            var_idx = var_names.index(var_name)
            arr = tile.data[var_idx].copy().astype(float)
            im = ax.imshow(arr, extent=extent, origin="lower", aspect="auto", cmap="viridis")
            ax.set_title(f"{band} / {param_labels[param]}", fontsize=8)
            ax.set_xlabel("Lon (°)", fontsize=6)
            ax.set_ylabel("Lat (°)", fontsize=6)
            plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viirs_brdf] Saved: {out_path}")


def _make_aod_figure(tile: GridTile, out_path: Path) -> None:
    """Save a single-panel merged AOD map."""
    import matplotlib.pyplot as plt

    bbox = tile.bounds
    extent = [tile.lons.min(), tile.lons.max(), tile.lats.min(), tile.lats.max()]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        tile.data.astype(float),
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="YlOrBr",
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        f"Merged AOD at 550 nm (AERDB_D3_GEOLEO)\n"
        f"Tile: lat [{bbox.lat_min:.1f}, {bbox.lat_max:.1f}]  "
        f"lon [{bbox.lon_min:.1f}, {bbox.lon_max:.1f}]  "
        f"({tile.data.shape[1]}×{tile.data.shape[0]} px)"
    )
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="AOD (unitless)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viirs_aod] Saved: {out_path}")


def _make_gridded_multivar_figure(tile: GridTile, out_path: Path, reader_key: str, suptitle: str) -> None:
    """Save a grid of panels, one per variable, for a rasterized multi-var tile.

    Used for the CERES SSF and CLDPIX readers, whose tiles carry many variables
    rasterized from swath/point data onto the 2° tile grid. Categorical
    variables (those with ``n_categories`` set) use a discrete colormap;
    continuous variables use a sequential one. Panels whose variable is entirely
    fill in this tile are explicitly annotated, so an empty panel reads as
    "no valid data here" rather than as a rendering failure.
    """
    import matplotlib.pyplot as plt

    variables = ReaderRegistry.get(reader_key).VARIABLES
    n = len(variables)
    ncol = 3
    nrow = math.ceil(n / ncol)

    bbox = tile.bounds
    extent = [tile.lons.min(), tile.lons.max(), tile.lats.min(), tile.lats.max()]

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(
        f"{suptitle}\n"
        f"Tile: lat [{bbox.lat_min:.1f}, {bbox.lat_max:.1f}]  "
        f"lon [{bbox.lon_min:.1f}, {bbox.lon_max:.1f}]  "
        f"({tile.data.shape[2]}×{tile.data.shape[1]} px)",
        fontsize=12,
    )
    for i, var_spec in enumerate(variables):
        ax = axes[i]
        arr = tile.data[i].astype(float)
        n_valid = int(np.isfinite(arr).sum())
        # Categorical variables get a discrete qualitative colormap; continuous
        # variables a perceptually-uniform sequential one.
        cmap = "tab20" if var_spec.n_categories is not None else "viridis"
        im = ax.imshow(arr, extent=extent, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(f"{var_spec.name}  ({n_valid} cells)", fontsize=8)
        ax.set_xlabel("Lon (°)", fontsize=6)
        ax.set_ylabel("Lat (°)", fontsize=6)
        if n_valid == 0:
            # Make a fully-fill panel unambiguous rather than a blank white box.
            ax.text(
                0.5,
                0.5,
                "all fill\n(no valid data\nin this tile)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="gray",
            )
        else:
            plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    # Hide any unused subplot axes.
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{reader_key}] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-reader load + report orchestration
# ---------------------------------------------------------------------------


def _run_reader(
    reader_key: str,
    file_path: Path,
    lat_center: float,
    lon_center: float,
    output_dir: Path,
    no_plots: bool,
) -> bool:
    """Load a single tile, print stats, and optionally save a plot.

    Returns True on success, False if an error occurs.
    """
    print(f"\n{'=' * 60}")
    print(f"  Loading: {reader_key.upper()}  ({file_path.name})")
    print(f"{'=' * 60}")

    try:
        reader_cls = ReaderRegistry.get(reader_key)
    except KeyError:
        print(f"  ERROR: reader '{reader_key}' not found in registry.")
        return False

    reader: GriddedDataReader = reader_cls(file_path)

    # Convert the center point to a TileKey.
    key = latlon_to_tile_key(reader_key, lat_center, lon_center)
    bbox = GriddedDataReader._tile_key_to_bbox(key)
    print(f"  Tile key : {key}")
    print(f"  Bbox     : {bbox}")

    try:
        tile = reader.load_tile(key)
    except ImportError as exc:
        print(f"  SKIPPED — library not available:\n    {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR loading tile:\n    {type(exc).__name__}: {exc}")
        return False

    if tile.data.size == 0:
        print("  WARNING: No data in this tile for the requested bbox.")
        print("  Try a different --lat-center / --lon-center.")
        return False

    _print_tile_summary(tile, reader_key)

    if not no_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("  matplotlib not installed — skipping visualization.")
            return True

        out_file = output_dir / f"{reader_key}_tile_verification.png"
        dispatch = {
            "igbp": _make_igbp_figure,
            "nise": _make_nise_figure,
            "era5": _make_era5_figure,
            "viirs_cloud": _make_viirs_cloud_figure,
            "viirs_brdf": _make_brdf_figure,
            "viirs_aod": _make_aod_figure,
            "ssf": lambda t, p: _make_gridded_multivar_figure(t, p, "ssf", "CERES SSF (rasterized)"),
            "cldpix": lambda t, p: _make_gridded_multivar_figure(t, p, "cldpix", "CERES CLDPIX (rasterized)"),
        }
        if reader_key in dispatch:
            dispatch[reader_key](tile, out_file)

    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_readers",
        description=textwrap.dedent("""\
            Verify footprint matching readers on real ancillary files.
            Supply any combination of reader file paths. Each reader loads
            the 2° tile containing --lat-center / --lon-center, prints
            statistics, and saves a PNG visualization.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--igbp", type=Path, metavar="FILE", help="MODIS MCD12Q1 HDF4 file")
    parser.add_argument("--nise", type=Path, metavar="FILE", help="NISE HDF-EOS4 file (*.HDFEOS)")
    parser.add_argument("--era5", type=Path, metavar="FILE", help="ERA5 NetCDF4 file with u10/v10 variables")
    parser.add_argument("--viirs", type=Path, metavar="FILE", help="VIIRS CLDPROP_D3 NetCDF4 file")
    parser.add_argument("--brdf", type=Path, metavar="FILE", help="VIIRS VJ143C1 BRDF HDF5 file")
    parser.add_argument("--aod", type=Path, metavar="FILE", help="AERDB_D3_GEOLEO merged AOD NetCDF4 file")
    parser.add_argument("--ssf", type=Path, metavar="FILE", help="CERES SSF (or FLASHFlux) NetCDF4 file")
    parser.add_argument("--cldpix", type=Path, metavar="FILE", help="CERES CLDPIX NetCDF4 file")
    parser.add_argument(
        "--lat-center", type=float, default=45.0, metavar="DEG", help="Tile center latitude in degrees (default: 45.0)"
    )
    parser.add_argument(
        "--lon-center",
        type=float,
        default=-90.0,
        metavar="DEG",
        help="Tile center longitude in degrees (default: -90.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Directory for PNG output (default: current directory)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Print statistics only; skip matplotlib visualization")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Map CLI argument names → registry keys
    requested: dict[str, Path | None] = {
        "igbp": args.igbp,
        "nise": args.nise,
        "era5": args.era5,
        "viirs_cloud": args.viirs,
        "viirs_brdf": args.brdf,
        "viirs_aod": args.aod,
        "ssf": args.ssf,
        "cldpix": args.cldpix,
    }

    if not any(requested.values()):
        parser.error(
            "No reader files supplied. Provide at least one of: "
            "--igbp, --nise, --era5, --viirs, --brdf, --aod, --ssf, --cldpix"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCenter point: lat={args.lat_center:.2f}°  lon={args.lon_center:.2f}°")
    print(f"Tile size: {TILE_SIZE_DEG}° × {TILE_SIZE_DEG}°")
    print(f"Output dir: {args.output_dir.resolve()}")

    results: dict[str, bool] = {}
    for reader_key, file_path in requested.items():
        if file_path is None:
            continue
        if not file_path.exists():
            print(f"\n  ERROR [{reader_key}]: file not found: {file_path}")
            results[reader_key] = False
            continue
        results[reader_key] = _run_reader(
            reader_key=reader_key,
            file_path=file_path,
            lat_center=args.lat_center,
            lon_center=args.lon_center,
            output_dir=args.output_dir,
            no_plots=args.no_plots,
        )

    print(f"\n{'─' * 60}")
    print("  Summary")
    print(f"{'─' * 60}")
    for reader_key, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {reader_key:<15s}  {status}")
    print()

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
