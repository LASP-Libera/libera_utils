# Libera Science Data Processing
Science data processing algorithms for the Libera mission.


## Further Documentation
- [Development Environment Setup](doc/dev_environment_setup.md)
- [Testing](doc/testing.md)
- [Release Process and Distribution](doc/build_release.md)
- [Package Configuration](doc/configuration.md)
- [SPICE Usage](doc/spice.md)
- [Docker (Including Nexus)](doc/docker.md)
- [Database Usage](doc/database.md)


## Installation from LASP PyPI
Note: This only works for officially released versions.
```bash
# Inside the LASP VPN only
pip install libera-sdp --index https://artifacts.pdmz.lasp.colorado.edu/repository/lasp-pypi/simple

# From any whitelisted IP (including any IP inside the LASP VPN)
pip install libera-sdp --index https://lasp.colorado.edu/repository/lasp-pypi/simple
```


## Basic Usage


### Command Line Interface
Depending on how you have install `libera_sdp`, your CLI runner may vary. The commands below assume that your 
virtual environment's `bin` directory is in your `PATH`. If you are developing the package, you may
want to use `poetry run` to run CLI commands.

#### Top Level Command `sdp`
```shell
sdp [--version] [-h]
```

#### Sub-Command `sdp make-kernel jpss-spk`
```shell
sdp make-kernel jpss-spk [-h] [--outdir OUTDIR] [--overwrite] packet_data_filepaths [packet_data_filepaths ...]
```


#### Sub-Command `sdp make-kernel jpss-ck`
```shell
sdp make-kernel jpss-ck [-h] [--outdir OUTDIR] [--overwrite] packet_data_filepaths [packet_data_filepaths ...]
```


#### Sub-Command `sdp make-kernel azel-ck`
Not yet implemented
```shell
sdp make-kernel azel-ck [-h]
```
