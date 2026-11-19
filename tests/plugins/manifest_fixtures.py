"""Plugin module for mocking S3 buckets"""

import os
import zipfile
from pathlib import Path

import pytest
from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier, ManifestType, ProcessingStepIdentifier
from libera_utils.footprint_matching.footprint_match_algorithm import (
    RUNNER_CONFIGS,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.manifest import PROCESSING_STEP_ID_CONFIGURATION_KEY, Manifest
from libera_utils.io.smart_open import smart_copy_file
from tests.plugins.footprint_matching_fixtures import L1B_DAY, write_cldpix_on_day

# The name a granule of each FMATCH reader's product is staged under for a run: a name in the product's ingested form,
# which the runner selects the granule by. The test data's upstream names carry a "_subset" suffix and its ERA5 files are
# not in the SDC's ERA5 form, so neither parses. CLDPIX is named for L1B_DAY, onto which it is dated, and IGBP is staged
# as its version archive, holding the tile under FMATCH_STAGED_IGBP_TILE_NAME.
FMATCH_STAGED_GRANULE_NAMES: dict[DataProductIdentifier, str] = {
    DataProductIdentifier.anc_ceres_noaa20_cldpix: "CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2026061115.nc",
    DataProductIdentifier.anc_era5_pressure_level: (
        "ECMWF_ERA5-PRESSURE-LEVEL_ERA5_20260611T000000_20260611T000000_R20260812T161100000000.nc"
    ),
    DataProductIdentifier.anc_era5_single_level: (
        "ECMWF_ERA5-SINGLE-LEVEL_ERA5_20260611T000000_20260611T000000_R20260812T161100000000.nc"
    ),
    DataProductIdentifier.anc_ceres_noaa20_ssf: "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026061100.nc",
    DataProductIdentifier.anc_modis_igbp: "MODIS-IGBP_061_20261020_R20261020T120000000000.zip",
    DataProductIdentifier.anc_nsidc_ice_snow: "NISE_AMSR2_20260611.HDFEOS",
    DataProductIdentifier.anc_viirs_noaa20_aod: "AERDB_D3_VIIRS_NOAA20.A2026162.021.2026166001045.nc",
    DataProductIdentifier.anc_viirs_noaa20_brdf: "VJ143C1.A2026162.002.2026175014229.h5",
    DataProductIdentifier.anc_viirs_noaa20_cloud_properties: "CLDPROP_D3_VIIRS_NOAA20.A2026162.011.2026166000752.nc",
}
FMATCH_STAGED_IGBP_TILE_NAME = "MCD12Q1.A2025001.h21v07.061.2026293135651.hdf"


@pytest.fixture
def stage_fmatch_ancillary_granules(tmp_path_factory, fmatch_real_ancillary_granules):
    """Factory staging the ancillary granule of every reader active for an FMATCH mode, named as the runner selects it.

    The returned function takes the directory to stage in (local or S3), the mode, and a keyword-only ``empty``. For
    the product of each active reader it writes one granule, the reader's real sample or an empty file when ``empty``
    is true, under the product's name in ``FMATCH_STAGED_GRANULE_NAMES``; IGBP's is a version archive holding the
    tile, and the real CLDPIX sample is dated onto ``L1B_DAY`` with ``write_cldpix_on_day``. Every real granule then
    covers the synthetic L1B input's day, so a run on it passes the granule date check. It returns the staged path by
    product.
    """

    def _stage(
        directory: Path | S3Path, mode: OperationalMode, *, empty: bool = False
    ) -> dict[DataProductIdentifier, Path | S3Path]:
        build_dir = tmp_path_factory.mktemp("staged_granules")
        if isinstance(directory, Path):
            directory.mkdir(parents=True, exist_ok=True)
        staged: dict[DataProductIdentifier, Path | S3Path] = {}
        for reader_key in sorted(ReaderRegistry.get_readers_for_mode(mode)):
            product_id = ReaderRegistry.get(reader_key).PRODUCT_ID
            if product_id in staged:
                continue
            source = None if empty else fmatch_real_ancillary_granules[reader_key]
            built = build_dir / FMATCH_STAGED_GRANULE_NAMES[product_id]
            if product_id is DataProductIdentifier.anc_modis_igbp:
                with zipfile.ZipFile(built, "w") as archive:
                    archive.writestr(FMATCH_STAGED_IGBP_TILE_NAME, b"" if source is None else source.read_bytes())
            elif source is None:
                built.touch()
            elif product_id is DataProductIdentifier.anc_ceres_noaa20_cldpix:
                write_cldpix_on_day(source, built, L1B_DAY)
            else:
                built.write_bytes(source.read_bytes())
            staged[product_id] = smart_copy_file(built, directory / built.name)
        return staged

    return _stage


@pytest.fixture
def generate_input_manifest_local(tmp_path, test_data_path):
    """Generating test manifest from the data in test_data

    The returned function takes the files to list, relative to ``test_data`` or absolute, and an optional keyword-only
    ``configuration`` dict for the manifest. It writes the manifest under ``tmp_path / "processing"``, so it can be
    called once per test.
    """

    def _generate_input_manifest_local_with_files(*filenames, configuration=None):
        files = []
        for filename in filenames:
            files.append(test_data_path / filename)

        input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration=configuration or {})
        input_manifest.add_files(*files)

        os.mkdir(tmp_path / "processing")
        input_manifest_file_path = input_manifest.write(out_path=tmp_path / "processing")

        return input_manifest_file_path

    return _generate_input_manifest_local_with_files


@pytest.fixture
def generate_input_manifest_s3(test_data_path, create_mock_bucket, write_file_to_s3):
    """Generating test manifest from the data in test_data

    The returned function takes the files to upload and list, relative to ``test_data``, and an optional keyword-only
    ``configuration`` dict for the manifest.
    """

    def _generate_input_manifest_s3_with_files(*filenames, configuration=None):
        r_bucket = create_mock_bucket()

        input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration=configuration or {})

        for filename in filenames:
            s3_file_path = f"s3://{r_bucket.name}/{filename}"
            local_path = test_data_path / filename
            write_file_to_s3(local_path, s3_file_path)
            input_manifest.add_files(s3_file_path)

        d_bucket = create_mock_bucket()
        input_manifest_file_path = input_manifest.write(out_path=f"s3://{d_bucket.name}/processing")

        return input_manifest_file_path

    return _generate_input_manifest_s3_with_files


@pytest.fixture
def fmatch_input_manifest(stage_fmatch_ancillary_granules):
    """Factory writing an FMATCH input manifest as a processing step hands it to the container.

    The returned function takes the processing step, the input products and the directory (local or S3) of the run's
    inputs. It copies the input products into that directory, stages there the real ancillary granule of every reader
    active for the step's mode with ``stage_fmatch_ancillary_granules``, lists them all in ``files`` and names the step
    in ``configuration["processing_step_id"]``. It returns the written manifest's path.
    """

    def _write(
        processing_step_id: ProcessingStepIdentifier, *input_products: Path, input_dir: Path | S3Path
    ) -> Path | S3Path:
        granules = stage_fmatch_ancillary_granules(input_dir, RUNNER_CONFIGS[processing_step_id].mode)
        manifest = Manifest(
            manifest_type=ManifestType.INPUT,
            configuration={PROCESSING_STEP_ID_CONFIGURATION_KEY: str(processing_step_id)},
        )
        manifest.add_files(*(smart_copy_file(path, input_dir / path.name) for path in input_products))
        manifest.add_files(*granules.values())
        return manifest.write(input_dir)

    return _write
