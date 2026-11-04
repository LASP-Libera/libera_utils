"""Product variable names derived from the snake_case names used inside libera_utils.

Libera data products name their variables in Camel_Case: each word capitalized, words
joined by underscores, acronyms in capitals. The convention comes from the CERES BDS files
and is the one the Libera L1B RAD-4CH and L1B CAM product definitions (libera_rad,
libera_cam) use, e.g. ``Solar_Zenith_Surface``, ``Filtered_Radiance_SW`` and
``Quality_Flag``.

Code inside libera_utils keeps snake_case names (reader specs, aggregation columns, scene
definitions). :func:`product_variable_name` derives the product name of an internal name,
and :func:`internal_variable_name` reads a product name back, so a product definition and
the code that fills it share one rule:

1. A ``variable`` row of ``PRODUCT_VARIABLE_NAMES_CSV`` renames an internal name outright,
   to match an L1B variable of the same quantity (``solar_zenith_angle`` →
   ``Solar_Zenith_Surface``).
2. Otherwise each ``_``-separated token is spelled from its ``token`` row when it has one
   (``viirs`` → ``VIIRS``, ``flashflux`` → ``FLASHFlux``), or with its first character upper
   cased, which leaves a token that starts with a digit (``2m``, ``055um``) as it is.

A token row changes only letter case, so outside the ``variable`` rows the internal name of
a product name is its lower-cased spelling. The table is read and validated when this module
is imported; a malformed table raises :class:`ValueError`.
"""

from __future__ import annotations

import csv
import re
from importlib.resources import files

#: Package resource holding the token spellings and whole-name renames.
PRODUCT_VARIABLE_NAMES_CSV = files("libera_utils.data") / "product_variable_names.csv"

_COLUMNS = ("kind", "internal", "product", "source")
_INTERNAL_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_INTERNAL_TOKEN = re.compile(r"[a-z][a-z0-9]*")
_PRODUCT_NAME = re.compile(r"[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*")


def _load_name_table() -> tuple[dict[str, str], dict[str, str]]:
    """Read and validate the token spellings and whole-name renames.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        The token spellings and the whole-name renames, each keyed by internal spelling.

    Raises
    ------
    ValueError
        If the table's columns are not ``kind,internal,product,source``, a row has another
        ``kind``, an empty ``source``, a malformed internal or product spelling, a token
        spelling that changes more than letter case, or repeats an internal spelling or a
        renamed product name.
    """
    tokens: dict[str, str] = {}
    renames: dict[str, str] = {}
    name = PRODUCT_VARIABLE_NAMES_CSV.name
    with PRODUCT_VARIABLE_NAMES_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != _COLUMNS:
            raise ValueError(f"{name} must have the columns {','.join(_COLUMNS)}, found {reader.fieldnames}.")
        for row in reader:
            where = f"{name}:{reader.line_num}"
            if None in row or any(row[column] is None for column in _COLUMNS):
                raise ValueError(f"{where}: expected {len(_COLUMNS)} columns; quote a value that contains a comma.")
            kind, internal, product, source = (row[column].strip() for column in _COLUMNS)
            if not source:
                raise ValueError(f"{where}: give the document or product that {product!r} comes from in 'source'.")
            if kind == "token":
                if not _INTERNAL_TOKEN.fullmatch(internal) or product.lower() != internal:
                    raise ValueError(
                        f"{where}: token {internal!r} -> {product!r} must be a lower-case token respelled "
                        "with letter case only."
                    )
                table = tokens
            elif kind == "variable":
                if not _INTERNAL_NAME.fullmatch(internal) or not _PRODUCT_NAME.fullmatch(product):
                    raise ValueError(
                        f"{where}: rename {internal!r} -> {product!r} must map a snake_case name to a "
                        "Camel_Case name such as 'Solar_Zenith_Surface'."
                    )
                if product in renames.values():
                    raise ValueError(f"{where}: {product!r} is already the product name of another internal name.")
                table = renames
            else:
                raise ValueError(f"{where}: kind {kind!r} must be 'token' or 'variable'.")
            if internal in table:
                raise ValueError(f"{where}: {kind} {internal!r} appears twice; keep one row.")
            table[internal] = product
    return tokens, renames


_TOKENS, _RENAMES = _load_name_table()
_RENAMED_FROM = {product: internal for internal, product in _RENAMES.items()}
_RENAME_TARGETS_LOWER = {product.lower(): product for product in _RENAMES.values()}


def product_variable_name(name: str) -> str:
    """Return the product variable name of an internal snake_case name.

    Parameters
    ----------
    name : str
        A snake_case name as used inside libera_utils, e.g. ``"era5_surface_wind_u10"``.

    Returns
    -------
    str
        The Camel_Case product name, e.g. ``"ERA5_Surface_Wind_U10"``.

    Raises
    ------
    ValueError
        If ``name`` is not snake_case (lower-case tokens joined by single underscores,
        starting with a letter), or if it is the lower-cased spelling of a renamed product
        name it is not the source of, which would give two internal names one product name.
    """
    if not _INTERNAL_NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} is not a snake_case internal name: use lower-case letters and digits in "
            "tokens joined by single underscores, starting with a letter."
        )
    if name in _RENAMES:
        return _RENAMES[name]
    if name in _RENAME_TARGETS_LOWER:
        target = _RENAME_TARGETS_LOWER[name]
        raise ValueError(
            f"{name!r} would be named {target!r}, which is already the product name of "
            f"{_RENAMED_FROM[target]!r}. Choose another internal name."
        )
    return "_".join(_TOKENS.get(token, token[:1].upper() + token[1:]) for token in name.split("_"))


def internal_variable_name(product_name: str) -> str:
    """Return the internal snake_case name of a product variable name.

    Parameters
    ----------
    product_name : str
        A product variable name, e.g. ``"VIIRS_BRDF_VIS_Fiso"``.

    Returns
    -------
    str
        The internal name whose :func:`product_variable_name` is ``product_name``, e.g.
        ``"viirs_brdf_vis_fiso"``.

    Raises
    ------
    ValueError
        If no internal name has ``product_name`` as its product name, i.e. the name does
        not follow the product naming rule (wrong letter case for a token, such as
        ``"Viirs_BRDF_VIS_Fiso"``, or an L1B spelling that has no rename row).
    """
    internal = _RENAMED_FROM.get(product_name, product_name.lower())
    try:
        expected = product_variable_name(internal)
    except ValueError as error:
        raise ValueError(f"{product_name!r} does not follow the product variable naming rule: {error}") from error
    if expected != product_name:
        raise ValueError(
            f"{product_name!r} does not follow the product variable naming rule; the name for "
            f"{internal!r} is {expected!r}. Spell it that way, or add a row to {PRODUCT_VARIABLE_NAMES_CSV.name}."
        )
    return internal
