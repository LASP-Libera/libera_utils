# Version Changes

## 5.10.1

- FEAT: WFOV SCI (APID 1040) L1A processing stitches complete SOP→EOP images onto `CAMERA_TIME`, stores compressed JPEG-LS payloads in `WFOV_IMAGE_BLOB` (`uint8`/`BLOB_BYTE` + `WFOV_IMAGE_BLOB_LENGTH`), and deduplicates packet-level `ICIE__WFOV_DATA` with `PACKET_IMAGE_ID` traceability.
- FEAT: Decode trailing 8-byte NAND footer metadata into `WFOV_TRAILING_FOOTER_*` variables; add file-level quality attrs `n_missing_sop_or_eop`, `n_bad_images`, and `n_complete_images`.
- FEAT: Add `WFOV_CRC_VALID` placeholder (`-1` = not validated) and `validate_wfov_image_crc` stub pending LIBSDC-747.
- BREAKING: `CAMERA_TIME` now contains one row per **complete** stitched image only (not every qualifying SOP). Removed redundant `WFOV_IMAGE_COMPLETE`.

## 5.10.0

- FEAT: Kernel generation applies the measured Libera opto-mechanical frame misalignments from OAV3 ground testing. The Az/El mechanism CKs now encode rotation about the measured axes of rotation, and the radiometer frames carry the measured line-of-sight boresight, so the generated kernels reflect the true instrument geometry instead of nominal placeholders. The three source unit vectors (`LIBERA_AZ_AOR_IN_STAND`, `LIBERA_EL_AOR_IN_STAND`, `LIBERA_EL0_Z_IN_STAND`) are stored as keywords in the frame kernel; the Az/El CK configs switch from Euler angles to quaternions, and `kernel_maker.add_mechanism_ck_quaternions` builds the per-sample rotations about the measured axes (applied after the encoder correction).

## 5.9.3

- FEAT: Az/El mechanism CKs now apply deterministic sinusoidal encoder-angle corrections during kernel generation, so the stored quaternions reflect the true mechanism rotation rather than the raw encoder readout. Corrections are applied to the filtered encoder angles in `create_kernel_from_l1a` (L1A telemetry is left unchanged), with matching forward/inverse helpers (`correct_azimuth`/`correct_elevation`, `uncorrect_azimuth`/`uncorrect_elevation`, and DataFrame-level `apply_encoder_corrections`/`reverse_encoder_corrections`)
- BUGFIX: Remove np.NaN as curryer no longer pins Numpy < 2

## 5.9.2

- FEAT: Expand the RBSP+VIIRS imager `DataProductIdentifier` set (`l2_unf_rad_imager`, `l2_nb_bb_imager_camtime`, `l2_toa_flux_imager`, and matching AUX scene-ID/FMATCH/ADM members)
- BUGFIX: rename camera/imager DPI and `ProcessingStepIdentifier` members for consistency with product strings: `l2_unf_cam`→`l2_unf_rad_cam`/`l2-unf-rad-cam`, `l2_cf_rad_time`/`l2_cf_rad`→`l2_cf_cam`/`l2-cf-cam`, `l2_cf_cam_time`/`l2_cf_cam`→`l2_cf_cam_camtime`/`l2-cf-cam-camtime`, `l2_nb_bb_cam_time`/`NB-BB-CAM-TIME`→`l2_nb_bb_cam_camtime`/`NB-BB-CAM-CAMTIME`/`l2-nb-bb-cam-camtime`, `l2_unf_imager`→`l2_unf_rad_imager`/`l2-unf-rad-imager`, `l2_nb_bb_imager`→`l2_nb_bb_imager_camtime`/`l2-nb-bb-imager-camtime`

## 5.9.1

- FEAT: Scene Identification now reports the property bin min/max bounds alongside the scene ID. For each footprint and classification variable, `identify_scenes` adds `scene_bin_{type}_{variable}_min`/`_max` variables giving the bounds of the matched scene's bin (unbounded sides and unmatched footprints reported as `NaN`). Controlled by the new `report_bin_bounds` flag (default `True`) on `FootprintData.identify_scenes` and `SceneDefinition.identify_and_update`; existing `scene_id_{type}` columns are unchanged.
- FEAT: Add `SceneDefinition.get_bin_bounds_for_scene_id` and `Scene.get_bin_bounds` to look up the property bin bounds for a given scene ID.

## 5.9.0

- BREAKING: Rename `DataLevel.ANC` to `DataLevel.AUX` and split ancillary archive routing into separate AUX and CAL archive buckets.
- BREAKING: Replace provisional L2/ANC `DataProductIdentifier` and `ProcessingStepIdentifier` values with the camera cloud-fraction and RBSP+VIIRS imager product sets for July delivery (renamed product strings, new AUX/L2 members, and updated L2 team IAM role mappings).
- DOCS: Update user docs for the new product/level identifiers.

## 5.8.6

- FEAT: Manual processing now runs through the SDC event bus instead of triggering Step Functions directly. `step-function-trigger` emits a `ManualProcessing` event for a single step, and a new `manual-processing` CLI submits arbitrary custom DAGs (or the default DAG) across one or more dates, with an optional `--verify` that polls the Coordination Table.
- FEAT: `ecr-upload` assumes the algorithm's per-team L2 Team Role for L2/ADM images so non-admin L2 developers can push to their ECR repos; non-L2 steps use the default/`--profile` session unchanged.

## 5.8.5

- FEAT: `s3-utils put` now performs manual SDC data ingest instead of a direct archive upload. It accepts multiple file paths, stages each file to the SDC Ingest Dropbox bucket, and emits a single `NewFilesAvailable` event to the SDC event bus; the SDC Data Ingester then handles archiving and metadata/data-availability records. Accepts L0 and data product filenames (manifests rejected).
- FEAT: Add `get_l2_team_role_session` (in `aws/utils.py`) which builds a boto session from a profile/`AWS_PROFILE` and assumes an L2 team role (default `L2Developer/LiberaUtils`), raising `ValueError` if the base profile is not permitted to assume it. `s3_put_cli_handler` now uses it as its session source.
- FEAT: Add `--verify`/`--timeout` to `s3-utils put`. With `--verify`, the CLI blocks after emitting the ingest event and polls (read-only) until each file is confirmed in its archive bucket and recorded in the File Metadata table (and, for non-L0 products, the Data Availability table), logging a per-file summary and raising on timeout (default 5 minutes). Adds `find_dynamodb_table_in_account_by_partial_name` to `aws/utils.py`.
- MAINT: Remove the unused `libera_utils/db/dynamodb_utils.py` module (stale write-side helpers and an unused `get_dynamodb_table`); the ingest-verification DynamoDB reads are inlined in `verify_ingestion`.

## 5.8.4

- FEAT: `KernelManager.load_naif_kernels` also loads an extended Earth PCK (`earth_*_predict.bpc` via `NAIF_EARTH_EXTENDED_PCK_REGEX`) so ITRF93 orientation is available beyond the short high-precision `earth_000101_*.bpc` window (required for JPSS geolocation and future-epoch processing).

## 5.8.3

- FEAT: Add NOM-HK array groups to stack waypoint status and sequence execution fields into indexed arrays (`ARRAY_128` and `ARRAY_8`)
- BUGFIX: Normalize Unicode packet string fields during aggregation to ensure deterministic byte sizing
- NOTE: NOM-HK L1A changes the shape and dtype of five variables (replacing per-index fields with indexed arrays). Treat this as breaking for external consumers of those variables; a patch bump was chosen during early development. Changed variables: ICIE**SW_FP_WP_ST_WP, ICIE**SW_SEQ_EXEC_BUF_OP,
  ICIE**SW_SEQ_EXEC_POS_OP, ICIE**SW_SEQ_ST_OP, ICIE\_\_SW_SEQ_STOP_CD_OP

## 5.8.2

- FEAT: Faster ICIE WFOV SCI (APID 1040) packet-to-L1A by refreshing `icie_xtce_tlm.xml` and combining the 972-field WFOV `aggregation_groups` block from `icie_wfov_sci` in `l1a_processing_configs.yml`.

## 5.8.1

- FEAT: Added alpha calibration combination constants for data products

## 5.8.0

- BREAKING: `KernelManager.load_libera_dynamic_kernels` accepts **only** a `Sequence[str | pathlib.Path | S3Path]` (manifest-ordered path list). Directory expansion, scalar `Path`, and scalar local `str` are not accepted (use `[path]` for a single file). Materialization is inlined with a single normalization pass per entry; `_materialize_dynamic_kernel_paths` / `_is_dynamic_sources_sequence` are removed from the dispatch path.
- BREAKING: Renamed `load_libera_dynamic_kernels` argument `dynamic_kernel_directory` to `dynamic_kernel_sources` (broader semantics; no compatibility alias).
- FEAT: `KernelFileCache` can materialize kernels from local filesystem paths (`Path` or non-HTTP `str`), with documented resolution rules for relative paths and stable cache freshness after copy.
- FEAT: `KernelManager` caches Libera static kernels under the versioned user cache via `KernelFileCache` after building them under the existing short temporary directory. When every required artifact is already cached (and within `cache_timeout_days`), static kernel creation is skipped and kernels are furnished from the cache, consistent with NAIF generic kernels.
- Improve `KernelFileCache` docstrings and `spice_utils` type annotations (including `ensure_spice` overloads and tighter return types on kernel helpers).

## 5.7.1

- BUGFIX: Fix packet parsing to l1a of datasets with deduplicated timestamps for final data assembly
- FEAT: Added `ground data` option for deduplication to allow ground data processing to continue
- BUGFIX: With `ground_data=True`, non-identical duplicate coordinates no longer raise when `verbose=False`; warnings only if `verbose=True`.
- BUGFIX: Duplicate-coordinate diagnostics no longer cite the source sequence counter when that coordinate is absent from the dataset slice (e.g. inside sample groups).

## 5.7.0

- FEAT: Added metadata writer to make UMM-G metadata files from NetCDF-4 data files

## 5.6.0

- FEAT: Add dimension validation to data product definition
- Support safer enforcement of product definition specs on data products
- Improve logging and warning visibility in data product writer

## 5.5.6

- FEAT: Adding support for APIDs 1000, 1002, 1035, 1043, and 1044 to support calibration and status telemetry packets

## 5.5.5

- CONFIG: Update the ICIE TLM XTCE to latest version

## 5.5.4

- BUGFIX: Implement KernelManager for Make_Kernel, furnishing necessary kernels
- REFACTOR: Port Make_Kernel and SPICE-based time conversions to spice_utils to avoid circular imports

## 5.5.2

- BUGFIX: Improve resiliency of JSON logging system for cloudwatch logs

## 5.5.1

- BUGFIX: Fixed the netcdf writing needed for L1B camera product and clarifying netcdf engine usage

## 5.5.0

- REFACTOR: Adjusted CLI interface for AWS tools to improve usability and consistency
- FEAT: Allows the usage of the profile argument in the cli tools for specifying AWS credentials profiles
- REFACTOR: Changing the s3 utilities for list and put to use the new constants tools more effectively

## 5.4.8

- FEAT: Add lookups between L1a, APID, and L1a product definitions

## 5.4.7

- FEAT: Update Space Packet Parser dependency to take advantage of packet filtering functionality

## 5.4.6

- BUGFIX: update xarray version due to breaking change in netcdf writing

## 5.4.5

- FEAT: Add enforcement of valid versioning in Filename classes

# 5.4.4

- FEAT: Allow passing dynamic product level attributes to data product writer

## 5.4.3

- FEAT: Allow adding the archive prefix to the path when calling write_libera_data_product

## 5.4.2

- Change xarray netcdf engine to h5netcdf because netcdf4 does not support writing to filelike objects (e.g. opened files).

## 5.4.1

- Added a KernelManager class to handle SPICE kernel caching and retrieval

## 5.4.0

- REFACTOR: Refactor kernel_maker to use L1a Dataset object. Packet-based kernel creation is still possible via `create_kernels_from_packets`, which runs L1a processing internally
- CONFIG: Update Curryer kernel configs to match Libera AXIS_SAMPLE packets fields
- Move the L1A processing configurations to YAML instead of Python objects
- BUGFIX: Fix bug in `packets.py::_drop_duplicates` to allow dropping based on coordinate or dimension

## 5.3.0

- FEAT: scene_id module alpha release capable of identifying scenes from footprint data in Ceres SSF files.

## 5.2.2

- Updated the Curryer version pin to use at least 0.2.0, which is when the problematic subdependencies were removed

## 5.2.1

- Added scene_id module for processing footprint data to produce datasets with assigned scene IDs according to libera_utils standard definitions.

## 5.2.0

- FEAT: Implement processing to L1A Dataset from packets
- CONFIG: Add full ICIE XTCE to Libera Utils package

## 5.1.0

- Added UMMGranule Pydantic class to validate datasets against the UMM-G schema
- Added methods to convert Libera datasets to UMMGranule instances

## 5.0.0

- FEAT: Add `write_libera_data_product` to `netcdf.py` for one-shot product writes from numpy arrays
- BREAKING: Remove L1A `ProcessingStepIdentifier` (node) enum constants
- BREAKING: Refactor `DataProductDefinition` (previously `DataProductConfig`) and product definition validation
- MAINT: Remove `hdf.py` and associated tests

## 4.0.0

- BREAKING: Removed L0 file chunking support: The chunk_number parameter and related methods have been removed from L0 data product identifiers
- BREAKING: Refactored identifier classes: Moved DataProductIdentifier and ProcessingStepIdentifier from libera_utils.aws.constants to libera_utils.constants
- BREAKING: Removed deprecated classes: ProductName, CkObject, and SpkObject enums have been removed and consolidated into DataProductIdentifier
- BREAKING: SPICE kernel filename consolidation: Merged separate SPICE kernel filename classes into LiberaProductFilename
- Enhanced DataProductIdentifier: Now includes embedded metadata with processing level information and simplified lookup methods
- Enhanced ProcessingStepIdentifier: Now includes embedded product relationships and improved level derivation
- New DataLevel enum: Added structured processing level enumeration with built-in archive bucket name mapping
- Filename validation: Updated L0 filename patterns to remove chunk number validation
- Improved type safety: Better type hints and validation throughout identifier classes
- Updated documentation: Refreshed user documentation to reflect new file naming conventions
- Added the Libera APID values we care about for science processing to the LiberaApid enum class

## 3.2.4

- Changed the SPICE kernel creation API
- Added the SPICE kernel creation Docker image
- Added the Geolocation Tier-1 integration test case and data files

## 3.2.3

- Improved ECR authentication issues with the `libera-utils ecr-upload` CLI command for working with SSO authentication
- Updated S3 list tool to print by default
- Improved tooling of the aws constants module for both DataProductIdentifier and ProcessingStepIdentifier
- Added step function and policy names to ProcessingStepIdentifier
- Updated constants for L2 data products
- Updated step function trigger tool with url output and naming improvements
- Exposed useful L2 tools at highest level of the `libera_utils` package

## 3.2.2

- Added dimension handling for data product configuration
- Added internal DataArray data storage matching to known dimensions
- Added jenkins file for publishing to PyPI
- Added contact listserv for project
- Added json and yaml linting to the pre-commit hooks
- Added prettier configuration for formatting markdown, json, and yaml files
- Added internal Dataset object to the Configurable NetCDF-4 metadata object
- Added writing of output file for the Configurable NetCDF-4 data product object

## 3.2.1

- Added automatic formatting to the repository using `pre-commit` hooks
- Added a list-serv for ci/cd output as part of the Jenkins files
- BUGFIX: Fixed filenaming changes from 3.2.0 that affected the CDK deployment and usage

## 3.2.0

- Added pydantic models for a Configurable NetCDF-4 metadata object
- Includes static metadata that is populated and known by these tools
- Includes dynamic metadata that must be provided by user of these tools
- Changed the naming scheme of the SPICE data files to include SPK or CK

## 3.1.1

- BUGFIX: Testing improvement for mocking docker image building

## 3.1.0

- Added Curryer (lasp-curryer) library dependency for SPICE kernel creation and geolocation
- Added static SPICE kernels and configuration files for geolocating NOAA-20 / CERES
- BREAKING: Removed geolocation submodule, replaced by interfaces within Curryer

## 3.0.0

- BREAKING: Removed support for python version 3.9 and 3.10
- BREAKING: Updated the aws constants ProcessingStepIdentifier dump method to be to_str_with_chunk_number
- Improved the standardization of CLI commands to call cli_handler functions that wrap the main functionality

## 2.5.2

- Added the s3-utils cli interface with subcommands put, list, and cp for ease of use s3 interactions
- The cli subcommand `libera-utils s3-utils put` will upload a file to the correct S3 archive bucket given an algorithm
- The cli subcommand `libera-utils s3-utils list` will list all files in a given S3 archive bucket for a given algorithm
- The cli subcommand `libera-utils s3-utils cp` will copy a file from one S3 archive bucket to another location
- Added the Libera Archive bucket naming pattern to the aws constants module
- Improved type hinting for the smart_open module
- Improved readability of cli testing
- Removing testing against python 3.9 and 3.10

## 2.5.1

- BUGFIX: missing pyyaml dependency that prevented usage of logutil module

## 2.5.0

- Reimplement Manifest class as a pydantic model and integrate with dependent code
- BREAKING: Remove the deprecated add_file_to_manifest method from Manifest class
- BREAKING: Rename outpath to out_path on the Manifest class write method

## 2.4.5

- Add `--ecr-image-tags` option to `libera-utils ecr-upload` CLI for tagging remote algorithm images in an ECR
- Add validation and serialization methods to the `DataProductIdentifier` and `ProcessingStepProductIdentifier` enums

## 2.4.4

- Allow overriding the standard docker config.json file with a minimal file for ECR uploading to
  prevent ECR upload permission failures from cached docker login credentials from CDK deployments

## 2.4.3

- Changes to ecr_upload to support programmatic building and pushing of Docker images
- Remove DynamoDB docs (moved to `libera_cdk`)

## 2.4.2

- BREAKING: Remove the `AnyFilename` polymorphic class. Please use `AbstractValidFilename.from_file_path()`

## 2.4.1

- Updating requirements of methods to use keyword arguments rather than positional arguments
- Adding ProcessingStepIdentifier and DataProductIdentifier to the filenaming classes
- Updating ecr names to work with the completion checker testing in libera_cdk

## 2.4.0

- Add properties to filenaming classes to retrieve `data_product_id` and `processing_step_id`
- Add ProcessingStepIdentifier and DataProductIdentifier standardization to be used by downstream repos

## 2.3.1

- Fix os.path.join bug in filenaming module that broke mocked S3 paths and also fix typehinting

## 2.3.0

- Create CLI tools for AWS ECR image upload and Step Function triggering
- Update manifest filenames to use ULID instead of timestamp for unique identifiers
- Change logutil configure_task_logging to optionally log JSON to console
- Allow configure_task_logging to optionally propagate DEBUG messages from specific loggers
- Update documentation for how the database is used in the Libera project in DynamoDB
- Create tools for DynamoDB in AWS for .pds files (CONS and PDS)
- Replace the use of PostgreSQL with DynamoDB for the Libera project

## 2.2.0

- Add AnyFilename polymorphic class
- Change filename of all products to a LiberaDataProductFilename that inherits from AnyFilename
- Update filenaming convention to be all capital letters
- Improve API for manifest module
- Add prefixing to Filename classes for predictable archive paths
- Add prefixing for manifest files for predictable and navigable paths in s3 buckets
- Update git to include lfs and move test data to lfs
- Improve database manager including caching improvements
- Improve smart_copy_file and bug fixes to smart_open testing
- Refactoring and improving pds ingest for database entries and integration testing in CDK
- Added handling of construction records and pds files appropriately when ingesting
  - This includes reading a construction record and removing the pds file entry for the construction record itself
- Improved testing of pds ingest and pds file orm models to more accurately reflect use cases
- Added output manifest creation from input manifest to match timestamps in filenames of input and output manifests
- Refactored pds ingest to use AnyPath objects for handling file locations
- Added error handling to pds ingest

## 2.1.1

- Update dependency specification to speed up dependency resolution wrt botocore/urllib3
- Improve database initialization to work with libera_cdk changes
- Fix bug in Dockerfile that incorrectly set the default entrypoint
- Add preliminary instrument kernel

## 2.1.0

- Improve API to Manifest and Manifest.add_files
- Add manifest filename enforcement to Manifest class
- Update filenaming conventions for product filenames and SPICE kernels
- Allow adding an s3 bucket/prefix as a basepath for filenames

## 2.0.1

- Remove the extras dependency spec because of the way SQLAlchemy imports models

## 2.0.0

- Add filenaming classes
- Add manifest file class
- Add construction record parser
- Update DB schema to store construction records
- Update kernel generation CLI to use manifest file pattern
- Shift database and spice related libraries to extras (not installed by default)
- Add smart_copy_file function that can copy files to and from S3 and filesystem locations transparently
- Remove HDF-EOS5 filehandling code
- Add quality flag classes
- Change license to BSD3

## 1.0.0

- Stub out project structure
- Add build and release processes to readme
- Switch to Poetry for project dependency configuration and build management
- Add geolocation module
- Add tools in spiceutil module for caching SPICE kernels from NAIF
- Add missing unit testing coverage
- Add spice.md documentation on how the package uses and manages SPICE kernels
- Add database tooling, dev database, and ORM setup
- Add smart_open for opening local or S3 objects
- Add logging utility functions for setting up application logging
