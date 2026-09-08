# Basic Usage

## Command Line Interface

The CLI is installed as an executable in your virtual environment during installation of `libera_utils`.

### Top Level Command `libera-utils`

This is the top level command that contains all the nested sub-commands.

```shell
usage: libera-utils [-h] [--version]
                    {make-kernel,ecr-upload,step-function-trigger,manual-processing,register-algorithm-image,s3-utils} ...

Libera SDC utilities CLI

options:
  -h, --help            show this help message and exit
  --version             print current version of the CLI

subcommands:
  sub-commands for libera-utils CLI

  {make-kernel,ecr-upload,step-function-trigger,manual-processing,register-algorithm-image,s3-utils}
    make-kernel         generate SPICE kernels from a manifest file
    ecr-upload          Upload a docker image to the ECR repository for a specific algorithm and register its
                        version(s)
    step-function-trigger
                        Manually trigger a single processing step for one applicable date
    manual-processing   Manually run a custom processing DAG (or the default DAG) for one or more applicable dates
    register-algorithm-image
                        Emit a NewAlgorithmImage event for an already-uploaded ECR image so the SDC Registrar
                        creates its versioned Batch job definition
    s3-utils            Utilities for working with S3 archives for processing steps

```

### Sub-Command `ecr-upload`

This is a tool to upload a docker image to AWS ECR. The image name and tag identify the local docker image while
the `--ecr-tags` option specifies the tags to apply to the image in ECR. If `--ecr-tags` is not provided, only the
`latest` tag is applied by default. If `--ecr-tags` is specified, include `latest` explicitly if it should also be
applied.

By default the console shows a concise summary of each push (start, a per-tag summary of layers pushed, and the
resulting image digest). Pass `-v`/`--verbose` for DEBUG-level output including per-layer Docker push detail.

#### Registering the uploaded image

Uploading an image to the ECR does not, by itself, make a specific algorithm version runnable in the SDC: the
processing step function resolves a requested version by looking for an AWS Batch **job definition** that references
the matching image. For this reason `ecr-upload` **always** registers the version(s) it uploads: after pushing, it
emits a `NewAlgorithmImage` event for each non-`latest` ECR tag pushed, so the SDC Registrar creates the
corresponding versioned Batch job definition automatically.

```{tip}
Because `ecr-upload` registers for you, you do **not** need to run the standalone
{ref}`register-algorithm-image <sub-command-register-algorithm-image>` command after an upload. Use the standalone
command only when the image was already uploaded previously (see that section for details).
```

Because `latest` is a moving pointer rather than a concrete version, it is never registered; if the only tag pushed
is `latest`, the command logs a warning and registers nothing. Add `--verify` to block until each registered Batch
job definition is confirmed created **and** its ECR image is confirmed present, waiting up to `--timeout` seconds
(default 300). Verification needs only read permissions.

```shell
usage: libera-utils ecr-upload [-h] [--image-tag IMAGE_TAG] [--ecr-tags ECR_TAGS [ECR_TAGS ...]]
                               [--ignore-docker-config] [-v] [--verify] [--timeout TIMEOUT]
                               [--profile PROFILE]
                               algorithm_name image_name

positional arguments:
  algorithm_name        Processing step identifier used to determine the ECR repository name
  image_name            Local image to upload, as `image-name:image-tag` (e.g. `my-image:1.2.3`). The
                        tag is required; latest is not assumed, so pass it explicitly if that is the
                        image you mean.

options:
  -h, --help            show this help message and exit
  --image-tag IMAGE_TAG
                        (DEPRECATED) The current tag of the local image that will be uploaded. Give the tag
                        as part of the image name instead (e.g. `my-image:1.2.3`). If both are given, they
                        must agree.
  --ecr-tags ECR_TAGS [ECR_TAGS ...]
                        Tags to apply in ECR. Default is latest.
  --ignore-docker-config
                        Ignore the standard docker config.json to bypass the credential store
  -v, --verbose         Enable DEBUG-level console logging, including per-layer Docker push detail. Without this,
                        console logging is at INFO (push start, a per-tag summary, and the resulting digest).
  --verify              After registering, block until each Batch job definition is confirmed created and its ECR
                        image is confirmed present. Requires only read permissions.
  --timeout TIMEOUT     Seconds to wait for registration verification when --verify is set. Default is 300 (5 minutes).
  --profile PROFILE     AWS profile name to use for the AWS session (ECR, EventBridge, Batch). If not set, the
                        default profile is used. The AWS region is taken from this profile's configuration.
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

The local image is named the same way `docker` names it: give the tag directly in the positional image
argument, as `image-name:image-tag`. The tag is **required**: algorithm images are built under an explicit
`docker build -t <name>:<version>`, so a local `latest` usually does not exist, and omitting the tag now fails
immediately with an explanatory error rather than dying later on a missing `image-name:latest`. If `latest` is
genuinely the image you want, pass `my-image:latest` explicitly. The `--image-tag` option is a deprecated way
of supplying the same tag separately; it still works, but if both are given they must agree or the command
errors out.

Example usage:

```shell
# Upload the locally built sfc-flux:1.2.3 image (registration of 1.2.3 happens automatically):
libera-utils ecr-upload l2-comp-flux recently-built-sfc-flux:1.2.3 --ecr-tags latest 1.2.3 --ignore-docker-config

# Upload, register, and verify the Batch job definition and ECR image in one step:
libera-utils ecr-upload l2-comp-flux recently-built-sfc-flux:1.2.3 --ecr-tags latest 1.2.3 --verify

# Deprecated equivalent of the first example, kept working for backwards compatibility:
libera-utils ecr-upload l2-comp-flux recently-built-sfc-flux --image-tag 1.2.3 --ecr-tags latest 1.2.3

# Errors out: no local tag given, and latest is not assumed.
libera-utils ecr-upload l2-comp-flux recently-built-sfc-flux --ecr-tags latest 1.2.3
```

To get a list of specific algorithm names allowed in this command, run `libera-utils ecr-upload -h`

(sub-command-register-algorithm-image)=

### Sub-Command `register-algorithm-image`

Emits a `NewAlgorithmImage` event for an ECR image that has **already been uploaded**, so the SDC Registrar creates
the corresponding versioned Batch job definition. This is the standalone equivalent of the registration that
`ecr-upload` performs automatically.

```{important}
If you are uploading the image now, just use `ecr-upload`, which registers the version(s) for you — there is no need
to run `register-algorithm-image` separately in that case. Use `register-algorithm-image` only when the image is
already in the ECR (for example, you need to (re)register a version without re-pushing the image).
```

Provide the `algorithm_name` and the concrete `algorithm_version` (the ECR image tag to register). Add `--verify`
to block until the Batch job definition is confirmed registered **and** the referenced ECR image is confirmed
present (up to `--timeout` seconds, default 300; read-only). The image-presence check runs even if a matching job
definition already exists, so you never register a job definition for an image that is not actually in the ECR.
`--image-digest` is optional and carried only for provenance — the job definition references the tag, not the
digest.

```shell
usage: libera-utils register-algorithm-image [-h] [--image-digest IMAGE_DIGEST] [--verify] [--timeout TIMEOUT]
                                             [--profile PROFILE]
                                             algorithm_name algorithm_version

positional arguments:
  algorithm_name        Processing step identifier used to determine the ECR repository name
  algorithm_version     The concrete ECR image tag to register (e.g. 1.2.3). The image must already be in ECR.

options:
  -h, --help            show this help message and exit
  --image-digest IMAGE_DIGEST
                        Optional image digest (sha256:...) carried for provenance; the job definition references
                        the tag.
  --verify              After emitting the event, block until the Batch job definition is confirmed registered.
                        Requires only read permissions.
  --timeout TIMEOUT     Seconds to wait for registration verification when --verify is set. Default is 300 (5 minutes).
  --profile PROFILE     AWS profile name to use for the AWS session (ECR, EventBridge, Batch). If not set, the
                        default profile is used. The AWS region is taken from this profile's configuration.
```

Example usage:

```shell
# Register (and verify) a version whose image is already in the ECR:
libera-utils register-algorithm-image l2-comp-flux 1.2.3 --verify
```

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
