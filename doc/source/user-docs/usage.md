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
                        ['l2-cloud-fraction', 'l2-ssw-toa', 'libera-adms', 'l2-ssw-surface-flux', 'l2-far-ir-toa-flux', 'l1c-unfiltered',
                        'spice-azel', 'spice-jpss', 'l1b-rad', 'l1b-cam']

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
