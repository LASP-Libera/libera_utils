# Basic Usage

## Command Line Interface
Depending on how you have installed `libera_utils`, your CLI runner may vary. The commands below assume that your 
virtual environment's `bin` directory is in your `PATH`. If you are developing the package, you may
need to use `poetry run` to run CLI commands.

### Top Level Command `libera-utils`
```shell
libera-utils [--version] [-h]
```

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

### Sub-Command `packet-ingest input-manifest.json`
```shell
libera-utils packet-ingest [-h] [packet_data_filepath]
```
