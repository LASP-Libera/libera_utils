"""Patch external-documentation links into the FMATCH demo notebooks' markdown cells.

Usage: python add_links.py <notebook path>

Each replacement must match exactly one markdown cell, exactly once. For notebook 06 a
markdown+code cell pair (product-scope chart) is inserted after the scope-table cell.
"""

import sys
from pathlib import Path

import nbformat

ATBD44 = "https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf"
ATBD45 = "https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.5.pdf"
CERES_DATA = "https://ceres.larc.nasa.gov/data/"
PYPROJ = "https://pyproj4.github.io/pyproj/stable/"
PYPROJ_GEOD = "https://pyproj4.github.io/pyproj/stable/api/geod.html"
NETCDF = "https://www.unidata.ucar.edu/software/netcdf/"
HDFEOS = "https://www.hdfeos.org/"

REPLACEMENTS = {
    "01_geometry_psf.ipynb": [
        (
            "(ATBD Eq. 4.4-1)",
            f"([CERES ATBD subsystem 4.4, Eq. 4.4-1]({ATBD44}))",
        ),
        (
            "onto the WGS84 ellipsoid",
            "onto the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System)",
        ),
        (
            "the pyproj/rotation work",
            f"the [pyproj]({PYPROJ})/rotation work",
        ),
        (
            "Everything below runs with no external data — inputs are synthesized inline.",
            "Everything below runs with no external data — inputs are synthesized inline.\n"
            "\n"
            "**Background**: the [Libera mission](https://lasp.colorado.edu/libera/) measures\n"
            "Earth's radiation budget; FMATCH pairs each radiometer footprint with imager and\n"
            "ancillary scene information, following the CERES SSF convolution heritage.",
        ),
    ],
    "02_readers_gridded.ipynb": [
        (
            "| `era5` | ERA5 single-level reanalysis (winds + 5 IMAGER fields) | NetCDF4 |",
            "| `era5` | [ERA5 single-level reanalysis](https://cds.climate.copernicus.eu/datasets/"
            f"reanalysis-era5-single-levels) (winds + 5 IMAGER fields) | [NetCDF4]({NETCDF}) |",
        ),
        (
            "| `era5_pressure` | ERA5 pressure-level fields (37 levels) | NetCDF4 |",
            "| `era5_pressure` | [ERA5 pressure-level fields](https://cds.climate.copernicus.eu/"
            "datasets/reanalysis-era5-pressure-levels) (37 levels) | NetCDF4 |",
        ),
        (
            "| `igbp` | IGBP land cover (MODIS, sinusoidal grid) | HDF4 |",
            "| `igbp` | [IGBP land cover](https://lpdaac.usgs.gov/products/mcd12q1v061/) (MODIS, "
            "[sinusoidal grid](https://modis-land.gsfc.nasa.gov/MODLAND_grid.html)) | "
            "[HDF4](https://www.hdfgroup.org/solutions/hdf4/) |",
        ),
        (
            "| `nise` | NSIDC NISE snow/ice (EASE-Grid, both hemispheres) | HDF-EOS4 |",
            "| `nise` | [NSIDC NISE snow/ice](https://nsidc.org/data/nise/versions/5) "
            "([EASE-Grid](https://nsidc.org/data/user-resources/help-center/guide-ease-grids), "
            f"both hemispheres) | [HDF-EOS4]({HDFEOS}) |",
        ),
        (
            "| `viirs_aod` | VIIRS Deep Blue aerosol (AERDB_D3_VIIRS_NOAA20) | NetCDF4 |",
            "| `viirs_aod` | [VIIRS Deep Blue aerosol](https://ladsweb.modaps.eosdis.nasa.gov/"
            "missions-and-measurements/products/AERDB_D3_VIIRS_NOAA20) (AERDB_D3_VIIRS_NOAA20) | NetCDF4 |",
        ),
        (
            "| `viirs_brdf` | VIIRS BRDF (VJ143C1) | HDF-EOS5 |",
            "| `viirs_brdf` | [VIIRS BRDF](https://lpdaac.usgs.gov/products/vj143c1v002/) (VJ143C1) | "
            f"[HDF-EOS5]({HDFEOS}) |",
        ),
        (
            "(`__init_subclass__` hook)",
            "([`__init_subclass__`](https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__) hook)",
        ),
        (
            "is a\ntemplate method:",
            "is a\n[template method](https://en.wikipedia.org/wiki/Template_method_pattern):",
        ),
        (
            "The heavy geospatial stack (`pyproj`, `pyhdf`,\n`h5py`) is behind the optional `fmatch` extra",
            f"The heavy geospatial stack ([`pyproj`]({PYPROJ}), [`pyhdf`](https://fhs.github.io/pyhdf/),\n"
            "[`h5py`](https://docs.h5py.org/en/stable/)) is behind the optional "
            "[`fmatch` extra](https://python-poetry.org/docs/pyproject/#extras)",
        ),
        (
            "conventions as\na real CDS download",
            "conventions as\na real [CDS](https://cds.climate.copernicus.eu/) download",
        ),
        (
            "including the *descending* latitude axis",
            "including the [*descending* latitude axis]"
            "(https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)",
        ),
    ],
    "03_readers_swath.ipynb": [
        (
            "| `ssf` | CERES SSF / FLASHFlux TOA+surface fluxes & clouds | 1-D footprint list (~20 km) |",
            f"| `ssf` | [CERES SSF]({CERES_DATA}) / [FLASHFlux]({CERES_DATA}) TOA+surface fluxes & clouds "
            "| 1-D footprint list (~20 km) |",
        ),
        (
            "| `viirs_cloud` | VIIRS cloud properties (D3 daily grid) | gridded L3 |",
            "| `viirs_cloud` | [VIIRS cloud properties](https://ladsweb.modaps.eosdis.nasa.gov/"
            "missions-and-measurements/products/CLDPROP_D3_VIIRS_NOAA20) (D3 daily grid) | gridded L3 |",
        ),
        (
            "**colatitude** (0° at the North Pole)",
            "[**colatitude**](https://en.wikipedia.org/wiki/Colatitude) (0° at the North Pole)",
        ),
    ],
    "04_tiling.ipynb": [
        (
            "byte-budgeted LRU cache",
            "byte-budgeted [LRU cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU)",
        ),
        (
            "the native HDF5 stack is not\n   thread-safe)",
            "the native HDF5 stack is [not thread-safe](https://docs.h5py.org/en/stable/faq.html))",
        ),
        (
            "a background worker loads\ntheir tiles while the main thread computes PSF weights for the current one.",
            "a [background worker](https://docs.python.org/3/library/threading.html) loads\n"
            "their tiles while the main thread computes PSF weights for the current one —\n"
            "blocking file I/O releases the "
            "[GIL](https://docs.python.org/3/glossary.html#term-global-interpreter-lock), "
            "so I/O and compute genuinely overlap.",
        ),
    ],
    "05_weighting_aggregation.ipynb": [
        (
            "reader contracts in, one scalar per product variable out.",
            "reader contracts in, one scalar per product variable out.\n"
            "\n"
            "The PSF model and the 75%/95% coverage thresholds follow the\n"
            f"[CERES SSF ATBD, subsystem 4.4]({ATBD44}).",
        ),
        (
            "surface ECEF geometry",
            "surface [ECEF](https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system) geometry",
        ),
        (
            "`weighted_median`",
            "[`weighted_median`](https://en.wikipedia.org/wiki/Weighted_median)",
        ),
    ],
    "06_product_definitions.ipynb": [
        (
            "## 1. Every definition loads and validates against the Pydantic schema",
            "## 1. Every definition loads and validates against the "
            "[Pydantic](https://docs.pydantic.dev/latest/) schema",
        ),
        (
            "| `FMATCH-IMAGER-FLASH` | radiometer footprints | FLASHFlux (low-latency SSF) |",
            f"| `FMATCH-IMAGER-FLASH` | radiometer footprints | [FLASHFlux]({CERES_DATA}) (low-latency SSF) |",
        ),
    ],
    "07_product_assembly.ipynb": [
        (
            "into an `xarray.Dataset` validated",
            "into an [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) validated",
        ),
        (
            "(`sunglint_angle`), aggregated",
            "([`sunglint_angle`](https://en.wikipedia.org/wiki/Sunglint)), aggregated",
        ),
        (
            "coverage drops toward the CERES 75%/95% thresholds (dashed lines)",
            f"coverage drops toward the [CERES 75%/95% thresholds]({ATBD44}) (dashed lines)",
        ),
    ],
    "08_camera_segmentation.ipynb": [
        (
            "estimated from the image's own\n   ground sample distance",
            "estimated from the image's own\n"
            "   [ground sample distance](https://en.wikipedia.org/wiki/Ground_sample_distance)",
        ),
        (
            "one batched pyproj call (`bounding_box_from_points_batch`)",
            f"one batched [pyproj `Geod`]({PYPROJ_GEOD}) call (`bounding_box_from_points_batch`)",
        ),
        (
            "the per-block corner geodesics",
            "the per-block corner [geodesics](https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid)",
        ),
    ],
    "09_runners.ipynb": [
        (
            "write the\n  conformant NetCDF product plus an **output manifest**",
            f"write the\n  conformant [NetCDF]({NETCDF}) product plus an **output manifest**",
        ),
        (
            "one Docker image, dispatched by entry point.",
            "one [Docker image](https://docs.docker.com/reference/dockerfile/), dispatched by "
            "[entry point](https://packaging.python.org/en/latest/specifications/entry-points/).",
        ),
    ],
    "10_scene_id_imager.ipynb": [
        (
            "scene classes (TRMM, ERBE, unfiltering)",
            "scene classes (TRMM, "
            "[ERBE](https://en.wikipedia.org/wiki/Earth_Radiation_Budget_Experiment), unfiltering)",
        ),
        (
            "used to\nselect ADMs.",
            f"used to\nselect [ADMs (angular distribution models)]({ATBD45}).",
        ),
        (
            "(FLASHFlux-derived, PR 3)",
            f"([FLASHFlux]({CERES_DATA})-derived, PR 3)",
        ),
    ],
}

SCOPE_CHART_MARKDOWN = """\
The chart splits each product's variable count into the **shared core** (variables
common to all five products — geolocation, viewing geometry, QA) and
**product-specific** variables (each family's matched data sources)."""

SCOPE_CHART_CODE = '''\
import matplotlib.pyplot as plt

var_sets = {name: set(d.variables) for name, d in definitions.items()}
shared_core = set.intersection(*var_sets.values())

products = list(definitions)
shared = [len(shared_core)] * len(products)
specific = [len(var_sets[p] - shared_core) for p in products]

fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(
    products,
    shared,
    height=0.55,
    color="#4477AA",
    edgecolor="white",
    linewidth=1,
    label=f"shared core ({len(shared_core)})",
)
ax.barh(
    products,
    specific,
    height=0.55,
    left=shared,
    color="#CCBB44",
    edgecolor="white",
    linewidth=1,
    label="product-specific",
)
for y, p in enumerate(products):
    ax.text(len(var_sets[p]) + 0.5, y, str(len(var_sets[p])), va="center", fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("variables in product definition")
ax.set_title("FMATCH product scope: shared core vs product-specific variables")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.show()'''


def apply(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    for old, new in REPLACEMENTS[path.name]:
        hits = [
            i
            for i, c in enumerate(nb.cells)
            if c.cell_type == "markdown" and old in "".join(c.source)
        ]
        if len(hits) != 1:
            raise SystemExit(f"{path.name}: pattern matched {len(hits)} cells: {old[:70]!r}")
        cell = nb.cells[hits[0]]
        text = "".join(cell.source)
        if text.count(old) != 1:
            raise SystemExit(f"{path.name}: pattern not unique in cell: {old[:70]!r}")
        cell.source = text.replace(old, new, 1)

    if path.name == "06_product_definitions.ipynb" and not any(
        "shared_core" in "".join(c.source) for c in nb.cells if c.cell_type == "code"
    ):
        anchor = next(
            i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "rows.append" in "".join(c.source)
        )
        nb.cells.insert(anchor + 1, nbformat.v4.new_markdown_cell(SCOPE_CHART_MARKDOWN))
        nb.cells.insert(anchor + 2, nbformat.v4.new_code_cell(SCOPE_CHART_CODE))

    nbformat.validate(nb)
    nbformat.write(nb, path)
    print(f"patched {path.name}: {len(REPLACEMENTS[path.name])} link edits")


if __name__ == "__main__":
    apply(Path(sys.argv[1]))
