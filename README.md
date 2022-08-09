# Libera Science Data Processing
Science data processing algorithms for the Libera mission.


## Further Documentation
- [Development Environment Setup](doc/source/developer-setup/dev_environment_setup.md)
- [Testing and Static Analysis](doc/source/developer-setup/testing.md)
- [Release Process and Distribution](doc/source/developer-setup/build_release.md)
- [Package Configuration](doc/source/developer-setup/configuration.md)
- [SPICE Usage](doc/source/developer-setup/spice.md)
- [Docker (Including Nexus)](doc/source/developer-setup/docker.md)
- [Database Usage](doc/source/developer-setup/database.md)
- [Logging Usage](doc/source/developer-setup/logging.md)


## Installation from LASP PyPI
Note: This only works for officially released versions.
```bash
# Inside the LASP VPN only
pip install libera-utils --extra-index-url https://artifacts.pdmz.lasp.colorado.edu/repository/lasp-pypi/simple

# From any whitelisted IP (including any IP inside the LASP VPN)
pip install libera-utils --extra-index-url https://lasp.colorado.edu/repository/lasp-pypi/simple
```


## Basic Usage


### Command Line Interface
Depending on how you have installed `libera_utils`, your CLI runner may vary. The commands below assume that your 
virtual environment's `bin` directory is in your `PATH`. If you are developing the package, you may
want to use `poetry run` to run CLI commands.

#### Top Level Command `libera-utils`
```shell
libera-utils [--version] [-h]
```

#### Sub-Command `libera-utils make-kernel jpss-spk`
```shell
libera-utils make-kernel jpss-spk [-h] [--outdir OUTDIR] [--overwrite] packet_data_filepaths [packet_data_filepaths ...]
```


#### Sub-Command `libera-utils make-kernel jpss-ck`
```shell
libera-utils make-kernel jpss-ck [-h] [--outdir OUTDIR] [--overwrite] packet_data_filepaths [packet_data_filepaths ...]
```


#### Sub-Command `libera-utils make-kernel azel-ck`
Not yet implemented
```shell
libera-utils make-kernel azel-ck [-h]
```
