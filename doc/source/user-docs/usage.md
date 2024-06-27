# Basic Usage

## Command Line Interface
The CLI is installed as an executable in your virtual environment during installation of `libera_utils`.

### Top Level Command `libera-utils`
This is the top level command that contains all the nested sub-commands. You can display the version or help text
directly from this top level command.
```shell
libera-utils [--version] [-h]
```

### Sub-Command `ecr-upload`
This is a tool to upload a docker image to AWS ECR. The image name and tag are required as arguments. 
The algorithm name is optional.
```shell
libera-utils ecr-upload [-h] image_name image_tag algorithm_name [--verbose]
```

Example usage: 
```shell
libera-utils ecr-upload my_l2_ssw_toa_docker_image latest l2_ssw_toa
```
For all specific algorithm names to use in this command, check the 
[AWS constants API here](../api-doc/generated/libera_utils.aws.constants.rst) module.


### Sub-Command `make-kernel jpss-spk`
```shell
libera-utils make-kernel jpss-spk [-h] [--outdir OUTDIR] [--overwrite] packet_data_filepaths [packet_data_filepaths ...]
```

### Sub-Command `make-kernel jpss-ck`
```shell
libera-utils make-kernel jpss-ck [-h] [--outdir OUTDIR] [--overwrite] packet_data_filepaths [packet_data_filepaths ...]
```

### Sub-Command `make-kernel azel-ck`
Not yet implemented
```shell
libera-utils make-kernel azel-ck [-h]
```
