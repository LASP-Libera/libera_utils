# Basic Usage

## Command Line Interface

The CLI is installed as an executable in your virtual environment during installation of `libera_utils`.

### Top Level Command `libera-utils`

This is the top level command that contains all the nested sub-commands.

```shell
usage: libera-utils [-h] [--version]
                    {make-kernel,ecr-upload,step-function-trigger,manual-processing,s3-utils} ...

Libera SDC utilities CLI

options:
  -h, --help            show this help message and exit
  --version             print current version of the CLI

subcommands:
  sub-commands for libera-utils CLI

  {make-kernel,ecr-upload,step-function-trigger,manual-processing,s3-utils}
    make-kernel         generate SPICE kernels from a manifest file
    ecr-upload          Upload docker image to ECR repository for a specific algorithm
    step-function-trigger
                        Manually trigger a single processing step for one applicable date
    manual-processing   Manually run a custom processing DAG (or the default DAG) for one or more applicable dates
    s3-utils            Utilities for working with S3 archives for processing steps

```

### Sub-Command `ecr-upload`

This is a tool to upload a docker image to AWS ECR. The image name and tag identify the local docker image while
the `--ecr-tags` option specifies the tags to apply to the image in ECR. If `--ecr-tags` is not provided, only the
`latest` tag is applied by default. If `--ecr-tags` is specified, include `latest` explicitly if it should also be
applied.

```shell
usage: libera-utils ecr-upload [-h] [--image-tag IMAGE_TAG]
                               [--ecr-tags ECR_TAGS [ECR_TAGS ...]]
                               [--ignore-docker-config] [--profile PROFILE]
                               algorithm_name image_name

positional arguments:
  algorithm_name        Processing step identifier used to determine the ECR repository name
  image_name            Image name to upload

options:
  -h, --help            show this help message and exit
  --image-tag IMAGE_TAG
                        Current tag of the local image. Default is latest.
  --ecr-tags ECR_TAGS [ECR_TAGS ...]
                        Tags to apply in ECR. Default is latest.
  --ignore-docker-config
                        Ignore the standard docker config.json to bypass the credential store
  --profile PROFILE     AWS profile name to use. If not set, the default profile is used.
```

Current L2 processing step identifiers include:

```text
l2-unf-rad-cam
l2-cf-cam
l2-cf-cam-camtime
l2-nb-bb-cam-camtime
l2-toa-flux-cam
l2-unf-rad-imager
l2-comp-flux
l2-nb-bb-imager-camtime
l2-toa-flux-imager
```

Example usage:

```shell
libera-utils ecr-upload l2-comp-flux recently-built-sfc-flux \
  --image-tag latest --ecr-tags latest --ignore-docker-config
```

To get a list of specific algorithm names allowed in this command, run `libera-utils ecr-upload -h`

### Sub-Command `make-kernel jpss-spk`

```shell
usage: libera-utils make-kernel jpss-spk [-h] --outdir OUTDIR [--overwrite] [-v] packet_data_filepaths [packet_data_filepaths ...]

positional arguments:
  packet_data_filepaths
                        paths to L0 packet files

options:
  -h, --help            show this help message and exit
  --outdir OUTDIR, -o OUTDIR
                        output directory for generated SPK
  --overwrite           force overwriting an existing kernel if it exists
  -v, --verbose         set DEBUG level logging output
```

### Sub-Command `make-kernel jpss-ck`

```shell
usage: libera-utils make-kernel jpss-ck [-h] --outdir OUTDIR [--overwrite] [-v] packet_data_filepaths [packet_data_filepaths ...]

positional arguments:
  packet_data_filepaths
                        paths to L0 packet files

options:
  -h, --help            show this help message and exit
  --outdir OUTDIR, -o OUTDIR
                        output directory for generated CK
  --overwrite           force overwriting an existing kernel if it exists
  -v, --verbose         set DEBUG level logging output
```

### Sub-Command `make-kernel azel-ck`

```shell
usage: libera-utils make-kernel azel-ck [-h] [--azimuth] [--elevation] --outdir OUTDIR [--overwrite] [--csv] [-v] packet_data_filepaths [packet_data_filepaths ...]

positional arguments:
  packet_data_filepaths
                        paths to L0 packet files

options:
  -h, --help            show this help message and exit
  --azimuth             generate ck for Azimuth
  --elevation           generate ck for Elevation
  --outdir OUTDIR, -o OUTDIR
                        output directory for generated CK
  --overwrite           force overwriting an existing kernel if it exists
  --csv                 the provided Az and El packet_data_filepaths are ASCII csv files instead of binary CCSDS
  -v, --verbose         set DEBUG level logging output (otherwise set by LIBSDP_STREAM_LOG_LEVEL)
```

### Sub-Command `step-function-trigger`

```shell
usage: libera-utils step-function-trigger [-h] [--verify] [--wait-time WAIT_TIME]
                                          [--profile PROFILE]
                                          algorithm_name applicable_day

positional arguments:
  algorithm_name        Processing step identifier to run
  applicable_day        Day of data to run. Format: YYYY-MM-DD

options:
  -h, --help            show this help message and exit
  --verify              Poll the Coordination Table to verify that the job was created
  --wait-time WAIT_TIME
                        Maximum verification wait in seconds. Default is 60.
  --profile PROFILE     AWS profile name to use. If not set, the default profile is used.
```

### Sub-Command `s3-utils`

Utilities for working with the SDC's S3 archives. The `--profile` option (or default boto authentication, e.g.
`AWS_PROFILE`) selects the AWS credentials used for all sub-commands. It must be supplied _before_ the sub-command,
e.g. `libera-utils s3-utils --profile my-profile put ...`.

```shell
usage: libera-utils s3-utils [-h] [--profile PROFILE] {put,ls,cp} ...

options:
  -h, --help         show this help message and exit
  --profile PROFILE  AWS profile name to use when accessing S3. If not set, the default profile is used.
```

#### Sub-Command `s3-utils put`

Stages one or more Libera data product files for ingest into the SDC. This does **not** write directly to an archive
bucket. Instead, each file is uploaded to the SDC Ingest Dropbox bucket and a single `NewFilesAvailable` event is
emitted to the SDC event bus. The SDC Data Ingester service then archives the files and creates the associated file
metadata and data availability records — exactly as it does for files produced by automated processing steps. The
command returns once the files are staged and the event is emitted; the ingest itself runs asynchronously, so it may
take a few minutes for files to appear in their archive bucket.

Each path must be a properly named Libera L0 or data product file (manifests and other filename types are rejected).

By default the command returns as soon as the files are staged and the event is emitted. Pass `--verify` to instead
block until each file is confirmed fully ingested — that is, present in its archive bucket, with a File Metadata
record and (for non-L0 data products) a Data Availability record. Verification needs only read permissions. Use
`--timeout` to control how long to wait (default 300 seconds); if any file is not fully ingested by then the command
logs a per-file summary and exits with an error.

```shell
usage: libera-utils s3-utils put [-h] [--verify] [--timeout TIMEOUT] file_path [file_path ...]

positional arguments:
  file_path          Path(s) to the file(s) to ingest. Each must be a properly named Libera L0 or data product file.

options:
  -h, --help         show this help message and exit
  --verify           After triggering ingest, block until each file is confirmed fully ingested, then report the result.
  --timeout TIMEOUT  Seconds to wait for ingestion verification when --verify is set. Default is 300 (5 minutes).
```

Example usage:

```shell
libera-utils s3-utils --profile my-profile put \
  LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc \
  LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc

# Stage one file and block until it is confirmed fully ingested (or 10 minutes elapse):
libera-utils s3-utils --profile my-profile put --verify --timeout 600 \
  LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc
```

#### Sub-Command `s3-utils ls`

Lists the files currently in the archive bucket for a given data product.

```shell
usage: libera-utils s3-utils ls [-h] product_name

positional arguments:
  product_name  The data product name string. Used to determine the S3 archive bucket name.

options:
  -h, --help    show this help message and exit
```

#### Sub-Command `s3-utils cp`

Copies an object between local and S3 locations (in either direction).

```shell
usage: libera-utils s3-utils cp [-h] [--delete] source_path dest_path

positional arguments:
  source_path  The current path to the object to retrieve
  dest_path    Destination path to save the object to

options:
  -h, --help   show this help message and exit
  --delete     If set, deletes files copied from source
```
