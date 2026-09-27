"""Unit tests for :mod:`libera_utils.io.auxiliary_filenaming`.

The corpus of positive filenames is taken verbatim from the ``example_data/`` tree (the real
vendor granules FMATCH consumes). Parsing inspects only the basename, so the tests pass bare
filename strings and do not depend on the (large, untracked) example files being present.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from libera_utils.constants import DataProductIdentifier
from libera_utils.io.auxiliary_filenaming import (
    AuxiliaryFilename,
    build_era5_canonical_filename,
    parse_auxiliary_filename,
    try_parse_auxiliary_filename,
)

# One representative real filename per convention, with every field the parser should extract.
# (filename, product_id, observation_date, version, production_time, source_variant)
_CASES = [
    (
        "MCD12Q1.A2024001.h21v07.061.2025205222320.hdf",
        DataProductIdentifier.auxiliary_igbp_mcd12q1,
        date(2024, 1, 1),
        "061",
        datetime(2025, 7, 24, 22, 23, 20),
        "h21v07",
    ),
    (
        "VJ143C1.A2026153.002.2026161161054.h5",
        DataProductIdentifier.auxiliary_viirs_brdf,
        date(2026, 6, 2),
        "002",
        datetime(2026, 6, 10, 16, 10, 54),
        None,
    ),
    (
        "VJ143C3.A2026153.002.2026161161054.h5",
        DataProductIdentifier.auxiliary_viirs_brdf_albedo,
        date(2026, 6, 2),
        "002",
        datetime(2026, 6, 10, 16, 10, 54),
        None,
    ),
    (
        "CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710.nc",
        DataProductIdentifier.auxiliary_viirs_cloud,
        date(2026, 5, 27),
        "011",
        datetime(2026, 5, 31, 0, 7, 10),
        "NOAA20",
    ),
    (
        "AERDB_D3_GEOLEO_Merged.A2020121.001.2024121023016.nc",
        DataProductIdentifier.auxiliary_viirs_aod,
        date(2020, 4, 30),
        "001",
        datetime(2024, 4, 30, 2, 30, 16),
        "GEOLEO_Merged",
    ),
    (
        "NISE_SSMISF18_20260111.HDFEOS",
        DataProductIdentifier.auxiliary_nise,
        date(2026, 1, 11),
        None,
        None,
        "SSMISF18",
    ),
    (
        "NISE_AMSR2_20260610.HDFEOS",
        DataProductIdentifier.auxiliary_nise,
        date(2026, 6, 10),
        None,
        None,
        "AMSR2",
    ),
    (
        "CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115.nc",
        DataProductIdentifier.auxiliary_ceres_ssf,
        date(2020, 4, 1),
        "alpha4",
        None,
        "NOAA20-FM6-VIIRS",
    ),
    (
        "CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc",
        DataProductIdentifier.auxiliary_ceres_cldpix,
        date(2020, 4, 10),
        "1P9test",
        None,
        "NOAA20-VIIRS",
    ),
    # ERA5 canonical names are Libera-assigned (applied by ingest), not vendor names -- see
    # build_era5_canonical_filename.
    (
        "ERA5-SINGLE-LEVEL_20260327.nc",
        DataProductIdentifier.auxiliary_era5_single_level,
        date(2026, 3, 27),
        None,
        None,
        None,
    ),
    (
        "ERA5-PRESSURE-LEVEL_20260327.nc",
        DataProductIdentifier.auxiliary_era5_pressure_level,
        date(2026, 3, 27),
        None,
        None,
        None,
    ),
]


class TestParseAuxiliaryFilename:
    @pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
    def test_parses_all_fields(self, case):
        # Targets the full parse for every convention; asserts every extracted field matches.
        filename, product_id, obs, version, production_time, source_variant = case
        parsed = parse_auxiliary_filename(filename)
        assert parsed.product_id is product_id
        assert parsed.observation_date == obs
        assert parsed.version == version
        assert parsed.production_time == production_time
        assert parsed.source_variant == source_variant

    def test_result_is_frozen_auxiliary_filename(self):
        # Targets the result type/immutability; asserts an AuxiliaryFilename that rejects mutation.
        parsed = parse_auxiliary_filename(_CASES[0][0])
        assert isinstance(parsed, AuxiliaryFilename)
        with pytest.raises(AttributeError):
            parsed.version = "999"  # type: ignore[misc]

    def test_path_is_preserved_and_only_basename_matters(self):
        # Targets path handling; asserts a full path parses the same as its basename and the path
        # is preserved on the result.
        filename = _CASES[0][0]
        parsed = parse_auxiliary_filename(f"/staged/inputs/igbp/{filename}")
        assert parsed.product_id is DataProductIdentifier.auxiliary_igbp_mcd12q1
        assert Path(parsed.path).name == filename

    def test_accepts_path_object(self):
        # Targets non-str input; asserts a pathlib.Path parses identically to the string form.
        parsed = parse_auxiliary_filename(Path(_CASES[0][0]))
        assert parsed.product_id is DataProductIdentifier.auxiliary_igbp_mcd12q1

    def test_aerdb_source_variant_distinguishes_wanted_granule(self):
        # The reader wants the single-sensor VIIRS_NOAA20 granule, not the cross-sensor merged one;
        # asserts both parse to the same product family but are distinguishable by source_variant.
        merged = parse_auxiliary_filename("AERDB_D3_GEOLEO_Merged.A2020121.001.2024121023016.nc")
        single = parse_auxiliary_filename("AERDB_D3_VIIRS_NOAA20.A2020121.001.2024121023016.nc")
        assert merged.product_id is single.product_id is DataProductIdentifier.auxiliary_viirs_aod
        assert merged.source_variant == "GEOLEO_Merged"
        assert single.source_variant == "VIIRS_NOAA20"

    @pytest.mark.parametrize(
        "filename",
        [
            "LIBERA_L1B_RAD-4CH_V0-5-6_20251120T175950_20251120T180020_R26190195454.nc",  # Libera product
            "68c76be0e3abb1bf23bcc15be5496e7b.nc",  # raw ERA5 CDS hash name
            "data_stream-oper_stepType-instant.nc",  # descriptive ERA5 name
            ".DS_Store",
            "MCD12Q1.A2024001.h21v07.061.2025205222320.nc",  # right stem, wrong extension
            "VJ143C2.A2026153.002.2026161161054.h5",  # unknown VJ143 band
            "not_a_product.txt",
        ],
    )
    def test_unrecognized_names_raise(self, filename):
        # Targets rejection of non-auxiliary names; asserts ValueError naming the offending file.
        with pytest.raises(ValueError, match="does not match any known auxiliary product"):
            parse_auxiliary_filename(filename)


class TestTryParseAuxiliaryFilename:
    def test_returns_parsed_for_known(self):
        # Targets the non-raising happy path; asserts it returns the same result as parse_*.
        parsed = try_parse_auxiliary_filename(_CASES[0][0])
        assert parsed is not None
        assert parsed.product_id is DataProductIdentifier.auxiliary_igbp_mcd12q1

    def test_returns_none_for_unknown(self):
        # Targets the manifest-scan use case; asserts a non-auxiliary name yields None, not an error.
        assert try_parse_auxiliary_filename("68c76be0e3abb1bf23bcc15be5496e7b.nc") is None
        assert try_parse_auxiliary_filename(".DS_Store") is None


class TestParserRegistryConsistency:
    """Every product the parser can emit must map to a real FMATCH reader (closes the loop)."""

    def test_every_parsed_product_maps_to_a_reader(self):
        # Ties the parser to the auxiliary product -> reader mapping in the footprint-matching
        # registry; asserts each parsed product_id resolves to a registered reader key.
        from libera_utils.footprint_matching.readers.registry import (
            AUXILIARY_PRODUCT_READERS,
            ReaderRegistry,
            reader_key_for_auxiliary_product,
        )

        registered = set(ReaderRegistry.list_readers())
        emitted = {parse_auxiliary_filename(case[0]).product_id for case in _CASES}
        # The corpus exercises every mapped auxiliary product (all 10 members, including both ERA5
        # canonical names); every emitted product must be in the mapping and resolve to a reader.
        assert emitted == set(AUXILIARY_PRODUCT_READERS)
        for product_id in emitted:
            assert reader_key_for_auxiliary_product(product_id) in registered


class TestBuildEra5CanonicalFilename:
    @pytest.mark.parametrize(
        ("product_id", "expected"),
        [
            (DataProductIdentifier.auxiliary_era5_single_level, "ERA5-SINGLE-LEVEL_20260327.nc"),
            (DataProductIdentifier.auxiliary_era5_pressure_level, "ERA5-PRESSURE-LEVEL_20260327.nc"),
        ],
    )
    def test_builds_canonical_name(self, product_id, expected):
        # Targets the ingest naming contract; asserts the exact canonical string.
        assert build_era5_canonical_filename(product_id, date(2026, 3, 27)) == expected

    @pytest.mark.parametrize(
        "product_id",
        [
            DataProductIdentifier.auxiliary_era5_single_level,
            DataProductIdentifier.auxiliary_era5_pressure_level,
        ],
    )
    def test_build_parse_round_trip(self, product_id):
        # The build helper and the parser are two ends of one contract; asserts a clean round-trip.
        parsed = parse_auxiliary_filename(build_era5_canonical_filename(product_id, date(2025, 12, 31)))
        assert parsed.product_id is product_id
        assert parsed.observation_date == date(2025, 12, 31)

    def test_rejects_non_era5_product(self):
        # Targets guard against misuse; asserts a non-ERA5 product raises ValueError.
        with pytest.raises(ValueError, match="is not an ERA5 auxiliary product"):
            build_era5_canonical_filename(DataProductIdentifier.auxiliary_nise, date(2026, 1, 1))
