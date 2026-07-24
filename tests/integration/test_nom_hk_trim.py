"""Integration tests for NOM-HK ObsID trimming against real L1A fixtures."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from libera_utils.l1a.nom_hk_trim import find_obsid_runs, write_trimmed_nom_hk_products
from libera_utils.obsids import NomHkObsidSource

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# TODO[LIBSDC-567]: Add integration coverage for WFOV/camera ObsID trimming once
# NOM-HK fixtures containing ICIE__SW_OBSID_WFOV cal events are available.

# Expected RAD cal runs in test_l1a_nom_hk_obsid_trim_subset
_EXPECTED_RAD_RUNS = {
    257: ("NOM-HK-SWC-405NM-TRIMMED", 236),
    385: ("NOM-HK-SOLAR-TOT-PRI-TRIMMED", 81),
    386: ("NOM-HK-SOLAR-LW-PRI-TRIMMED", 81),
}


def test_find_obsid_runs_on_fixture(test_l1a_nom_hk_obsid_trim_subset: Path):
    """Fixture contains the three expected RAD cal runs and no trim-eligible WFOV runs."""
    with xr.open_dataset(test_l1a_nom_hk_obsid_trim_subset) as ds:
        runs = find_obsid_runs(ds)

    rad_runs = [(spec, sl) for spec, sl in runs if spec.source is NomHkObsidSource.RAD]
    wfov_runs = [(spec, sl) for spec, sl in runs if spec.source is NomHkObsidSource.WFOV]
    assert wfov_runs == []
    assert {spec.obsid for spec, _ in rad_runs} == set(_EXPECTED_RAD_RUNS)

    for spec, pkt_slice in rad_runs:
        expected_product, expected_count = _EXPECTED_RAD_RUNS[spec.obsid]
        assert spec.trimmed_product is not None
        assert spec.trimmed_product.value == expected_product
        assert pkt_slice.stop - pkt_slice.start == expected_count


def test_write_trimmed_nom_hk_products_on_fixture(test_l1a_nom_hk_obsid_trim_subset: Path, tmp_path: Path):
    """End-to-end write produces one TRIMMED file per RAD cal ObsID with matching packet counts."""
    with xr.open_dataset(test_l1a_nom_hk_obsid_trim_subset) as ds:
        # Load so the Dataset outlives the closed file handle
        ds = ds.load()
        written = write_trimmed_nom_hk_products(ds, tmp_path, strict=True)

    assert len(written) == 3
    by_token = {path.path.name: path for path in written}
    for obsid, (product_token, expected_count) in _EXPECTED_RAD_RUNS.items():
        matches = [p for name, p in by_token.items() if product_token in name]
        assert len(matches) == 1, f"Expected one file for {product_token}, got {list(by_token)}"
        with xr.open_dataset(matches[0].path) as trimmed:
            assert trimmed.attrs["ProductID"] == product_token
            assert trimmed.sizes["PACKET"] == expected_count
            assert np.unique(trimmed["ICIE__SW_OBSID_RAD"].values).tolist() == [obsid]


def test_pad_obsids_do_not_produce_trimmed_products(test_l1a_nom_hk_obsid_trim_subset: Path, tmp_path: Path):
    """Non-cal pad ObsIDs in the fixture must not yield TRIMMED products."""
    with xr.open_dataset(test_l1a_nom_hk_obsid_trim_subset) as ds:
        present = set(np.unique(ds["ICIE__SW_OBSID_RAD"].values).tolist())
        ds = ds.load()
        written = write_trimmed_nom_hk_products(ds, tmp_path, strict=True)

    # Fixture includes non-cal pads (e.g. 128) that must not be trimmed
    assert present - set(_EXPECTED_RAD_RUNS)  # pads exist
    tokens = {p.path.name for p in written}
    assert len(tokens) == 3
    for name in tokens:
        assert any(tok in name for tok, _ in _EXPECTED_RAD_RUNS.values())
