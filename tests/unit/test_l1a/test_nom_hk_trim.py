"""Unit tests for libera_utils.l1a.nom_hk_trim."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.l1a.nom_hk_trim import (
    find_obsid_runs,
    get_trimmed_nom_hk_product_definition,
    trim_nom_hk_run,
    write_trimmed_nom_hk_products,
)
from libera_utils.obsids import NomHkObsidSource


def _synthetic_nom_hk(
    rad_obsids: list[int],
    wfov_obsids: list[int] | None = None,
    *,
    start: str = "2028-02-13T02:00:00",
) -> xr.Dataset:
    """Build a minimal NOM-HK-shaped Dataset for run-detection unit tests."""
    n = len(rad_obsids)
    if wfov_obsids is None:
        wfov_obsids = [0] * n
    assert len(wfov_obsids) == n
    times = np.datetime64(start) + np.arange(n) * np.timedelta64(1, "s")
    return xr.Dataset(
        {
            "ICIE__SW_OBSID_RAD": ("PACKET", np.asarray(rad_obsids, dtype=np.uint16)),
            "ICIE__SW_OBSID_WFOV": ("PACKET", np.asarray(wfov_obsids, dtype=np.uint16)),
        },
        coords={"PACKET_ICIE_TIME": ("PACKET", times)},
        attrs={
            "ProductID": "NOM-HK-DECODED",
            "algorithm_version": "5.10.0",
            "input_files": ["fake.bin"],
            "date_created": "2028-01-01T00:00:00+00:00",
        },
    )


class TestGetTrimmedNomHkProductDefinition:
    """Product-definition ProductID override."""

    def test_overrides_product_id(self):
        definition = get_trimmed_nom_hk_product_definition(DataProductIdentifier.l1a_icie_nom_hk_swc_405nm_trimmed)
        assert definition.attributes["ProductID"] == "NOM-HK-SWC-405NM-TRIMMED"

    def test_rejects_non_trimmed_product(self):
        with pytest.raises(ValueError, match="not a known TRIMMED"):
            get_trimmed_nom_hk_product_definition(DataProductIdentifier.l1a_icie_nom_hk_decoded)


class TestFindObsidRuns:
    """Contiguous ObsID run detection."""

    def test_rad_and_wfov_in_same_dataset(self):
        # RAD: 257 run, then pad, then 385; WFOV: 256 (darks) in the middle
        ds = _synthetic_nom_hk(
            rad_obsids=[128, 257, 257, 257, 128, 385, 385, 128],
            wfov_obsids=[0, 0, 0, 0, 256, 256, 0, 0],
        )
        runs = find_obsid_runs(ds)
        by_key = {(spec.source, spec.obsid): pkt_slice for spec, pkt_slice in runs}
        assert (NomHkObsidSource.RAD, 257) in by_key
        assert (NomHkObsidSource.RAD, 385) in by_key
        assert (NomHkObsidSource.WFOV, 256) in by_key
        assert by_key[(NomHkObsidSource.RAD, 257)] == slice(1, 4)
        assert by_key[(NomHkObsidSource.RAD, 385)] == slice(5, 7)
        assert by_key[(NomHkObsidSource.WFOV, 256)] == slice(4, 6)

    def test_science_obsid_not_returned(self):
        ds = _synthetic_nom_hk(rad_obsids=[136, 136, 136], wfov_obsids=[136, 136, 136])
        assert find_obsid_runs(ds) == []

    def test_rad_vs_wfov_256_are_distinct(self):
        ds = _synthetic_nom_hk(
            rad_obsids=[256, 256, 128],
            wfov_obsids=[256, 256, 0],
        )
        runs = find_obsid_runs(ds)
        products = {spec.trimmed_product for spec, _ in runs}
        assert DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed in products
        assert DataProductIdentifier.l1a_icie_nom_hk_darks_of_darks_trimmed in products

    def test_source_filter(self):
        ds = _synthetic_nom_hk(
            rad_obsids=[257, 257],
            wfov_obsids=[129, 129],
        )
        rad_only = find_obsid_runs(ds, source=NomHkObsidSource.RAD)
        assert len(rad_only) == 1
        assert rad_only[0][0].obsid == 257


class TestTrimNomHkRun:
    """Packet subsetting for one run."""

    def test_trims_packet_count(self):
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 257, 128])
        trimmed = trim_nom_hk_run(ds, slice(1, 4))
        assert trimmed is not None
        assert trimmed.sizes["PACKET"] == 3
        assert set(trimmed["ICIE__SW_OBSID_RAD"].values.tolist()) == {257}

    def test_empty_returns_none(self):
        ds = _synthetic_nom_hk(rad_obsids=[128, 128])
        assert trim_nom_hk_run(ds, slice(0, 0)) is None


class TestWriteTrimmedNomHkProducts:
    """High-level writer orchestration (write_libera_data_product mocked)."""

    def test_multi_run_same_obsid_warns_and_writes_twice(self, tmp_path, caplog):
        # Two disjoint 257 runs
        ds = _synthetic_nom_hk(rad_obsids=[257, 257, 128, 257, 257])
        mock_filename = MagicMock()
        mock_filename.path.name = "LIBERA_L1A_NOM-HK-SWC-405NM-TRIMMED_fake.nc"

        with (
            patch(
                "libera_utils.l1a.nom_hk_trim.write_libera_data_product",
                return_value=mock_filename,
            ) as mock_write,
            caplog.at_level("WARNING"),
        ):
            written = write_trimmed_nom_hk_products(ds, tmp_path, strict=False)

        assert len(written) == 2
        assert mock_write.call_count == 2
        assert any("appears in 2 disjoint runs" in r.message for r in caplog.records)

    def test_writes_distinct_products_for_rad_and_wfov(self, tmp_path):
        ds = _synthetic_nom_hk(
            rad_obsids=[257, 257],
            wfov_obsids=[256, 256],
        )
        mock_filename = MagicMock()
        mock_filename.path.name = "fake.nc"
        with patch(
            "libera_utils.l1a.nom_hk_trim.write_libera_data_product",
            return_value=mock_filename,
        ) as mock_write:
            write_trimmed_nom_hk_products(ds, tmp_path, strict=False)

        product_ids = {call.args[1].attrs["ProductID"] for call in mock_write.call_args_list}
        assert "NOM-HK-SWC-405NM-TRIMMED" in product_ids
        assert "NOM-HK-DARKS-OF-DARKS-TRIMMED" in product_ids
