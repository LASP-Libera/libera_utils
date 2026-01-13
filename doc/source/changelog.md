# Version Changes

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
