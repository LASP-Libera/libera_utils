# Basic Usage

## Command Line Interface

The CLI is installed as an executable in your virtual environment during installation of `libera_utils`.

### Top Level Command `libera-utils`

This is the top level command that contains all the nested sub-commands.

```shell
usage: libera-utils [-h] [--version] {make-kernel,ecr-upload,step-function-trigger} ...

Libera SDC utilities CLI

options:
  -h, --help            show this help message and exit
  --version             print current version of the CLI

subcommands:
  sub-commands for libera-utils CLI

  {make-kernel,ecr-upload,step-function-trigger}
    make-kernel         generate SPICE kernel from telemetry data
    ecr-upload          Upload docker image to ECR repository for a specific algorithm
    step-function-trigger
                        Manually trigger a specific step function

```

### Sub-Command `ecr-upload`

This is a tool to upload a docker image to AWS ECR. The image name and tag identify the local docker image while
the --ecr-image-tags option specifies the tags to apply to the image in the ECR (remote tags). If `--ecr-image-tags`
is not provided, only the `latest` tag is applied by default. If `--ecr-image-tags` is specified, you must include
`latest` explicitly.

```shell
usage: libera-utils ecr-upload [-h] [--ecr-image-tags ECR_IMAGE_TAGS [ECR_IMAGE_TAGS ...]] [--ignore-docker-config] image_name image_tag algorithm_name

positional arguments:
  image_name            Image name of image to upload (image-name:image-tag)
  image_tag             Image tag of image to upload (image-name:image-tag)
  algorithm_name        Algorithm name that matches an ECR repo name, inputs to names:
                        ['cal-rad', 'cal-cam', 'spice-azel', 'spice-jpss', 'l1b-rad', 'l1b-cam', 'int-footprint-scene-id',
                        'l2-cf-rad', 'l2-cf-cam', 'l2-unfiltered', 'l2-ssw-toa-osse', 'l2-ssw-toa-erbe', 'l2-ssw-toa-trmm',
                        'l2-ssw-toa-rt', 'l2-ssw-surface-flux', 'adm-binning']

options:
  -h, --help            show this help message and exit
  --ecr-image-tags ECR_IMAGE_TAGS [ECR_IMAGE_TAGS ...]
                        List of tags to apply to the uploaded image in the ECR (e.g. `--ecr-image-tags latest 1.3.4`) Note, latest is applied if this option is not set. If it is set, you must specify
                        latest if you want it tagged as such in the ECR.
  --ignore-docker-config
                        Ignore the standard docker config.json to bypass the credential store
```

Example usage:

```shell
libera-utils ecr-upload recently-built-ssw-sfc-flux latest l2-ssw-surface-flux --ecr-image-tags latest --ignore-docker-config
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
usage: libera-utils step-function-trigger [-h] [-w] [-v] algorithm_name applicable_day

positional arguments:
  algorithm_name        Algorithm name you want to run
  applicable_day        Day of data you want to rerun. Format of date: YYYY-MM-DD

options:
  -h, --help            show this help message and exit
  -w, --wait_for_finish
                        Block command line until step function completes (may be a long time)
  -v, --verbose         Prints out the result of the step_function_trigger run
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
