"""Integration tests for NOM-HK ObsID trimming against real L1A fixtures."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from libera_utils.l1a.nom_hk_trim import find_obsid_runs, write_trimmed_nom_hk_products
from libera_utils.obsids import NomHkObsidSource
from libera_utils.version import version as libera_utils_version

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# TODO[LIBSDC-567]: Add integration coverage for WFOV/camera ObsID trimming once
# NOM-HK fixtures containing ICIE__SW_OBSID_WFOV cal events are available.

# Expected RAD cal runs in test_l1a_nom_hk_obsid_trim_subset, as ObsID -> (family ProductID, packets).
# ObsIDs 385 and 386 are both solar-diffuser cals, so they share one TRIMMED family ProductID and are
# told apart by the ObsID field inside each file rather than by the filename.
_EXPECTED_RAD_RUNS = {
    257: ("NOM-HK-SWC-FAMILY-TRIMMED", 236),
    385: ("NOM-HK-SOLAR-FAMILY-TRIMMED", 81),
    386: ("NOM-HK-SOLAR-FAMILY-TRIMMED", 81),
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
    """End-to-end write produces one TRIMMED file per RAD cal ObsID run, stamped with its family."""
    with xr.open_dataset(test_l1a_nom_hk_obsid_trim_subset) as ds:
        # Load so the Dataset outlives the closed file handle
        ds = ds.load()
        written = write_trimmed_nom_hk_products(
            ds, tmp_path, source_product_filename=test_l1a_nom_hk_obsid_trim_subset, strict=True
        )

    assert len(written) == 3
    # Files are keyed by the ObsID they actually contain; two of them share a family ProductID
    by_obsid = {}
    for path in written:
        with xr.open_dataset(path.path) as trimmed:
            obsids = np.unique(trimmed["ICIE__SW_OBSID_RAD"].values).tolist()
            assert len(obsids) == 1, f"{path.path.name} covers more than one ObsID: {obsids}"
            assert obsids[0] not in by_obsid, f"ObsID {obsids[0]} was written more than once"
            by_obsid[obsids[0]] = {
                "ProductID": trimmed.attrs["ProductID"],
                "packets": trimmed.sizes["PACKET"],
                # NetCDF collapses a single-element string list to a scalar on round-trip.
                "input_files": np.atleast_1d(trimmed.attrs["input_files"]).tolist(),
                "algorithm_version": trimmed.attrs["algorithm_version"],
            }

    assert set(by_obsid) == set(_EXPECTED_RAD_RUNS)
    for obsid, (product_token, expected_count) in _EXPECTED_RAD_RUNS.items():
        assert by_obsid[obsid]["ProductID"] == product_token
        assert by_obsid[obsid]["packets"] == expected_count
        # Provenance points at the parent granule, not the L0 packet files it was decoded from
        assert by_obsid[obsid]["input_files"] == [test_l1a_nom_hk_obsid_trim_subset.name]
        assert by_obsid[obsid]["algorithm_version"] == libera_utils_version()
    # Both solar ObsIDs landed in the one solar family product, under distinct filenames
    solar_names = {path.path.name for path in written if "SOLAR-FAMILY" in path.path.name}
    assert len(solar_names) == 2


def test_pad_obsids_do_not_produce_trimmed_products(test_l1a_nom_hk_obsid_trim_subset: Path, tmp_path: Path):
    """Non-cal pad ObsIDs in the fixture must not yield TRIMMED products."""
    with xr.open_dataset(test_l1a_nom_hk_obsid_trim_subset) as ds:
        present = set(np.unique(ds["ICIE__SW_OBSID_RAD"].values).tolist())
        ds = ds.load()
        written = write_trimmed_nom_hk_products(
            ds, tmp_path, source_product_filename=test_l1a_nom_hk_obsid_trim_subset, strict=True
        )

    # Fixture includes non-cal pads (e.g. 128) that must not be trimmed
    assert present - set(_EXPECTED_RAD_RUNS)  # pads exist
    names = {p.path.name for p in written}
    assert len(names) == 3
    for name in names:
        assert any(tok in name for tok, _ in _EXPECTED_RAD_RUNS.values())
