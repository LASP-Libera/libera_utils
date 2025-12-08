# Making CK and SPK SPICE Kernels

The process for creating CK and SPK kernels that we generate (SC SPK, SC CK, Az CK, El CK) follows three different paths:

1. **Production:** Manifest --> L1A NetCDF --> L1A Dataset --> Curryer DataFrame --> Kernel
2. **End to End Testing:** Packets --> L1A Dataset --> Curryer DataFrame --> Kernel
3. **Simulation Testing:** Simulated CSV --> Curryer DataFrame --> Kernel

## Production

In our production system, L1A processing is done before kernel generation occurs. Kernel
generation starts from a manifest containing a single L1A NetCDF data product containing decoded
packets in a particular structure, determined in the configuration of the L1A pipeline.

Steps:

1. Read manifest file
2. Read L1A NetCDF product into L1A xr.Dataset
3. Extract kernel data from Dataset into pd.DataFrame for Curryer
4. Call Curryer kernel maker with config file and input DataFrame

## End to End Testing

For end to end testing, we often start directly from packet data to generate a kernel.

Steps:

1. Process packet data to L1A xr.Dataset
2. Extract kernel data from Dataset into pd.DataFrame for Curryer
3. Call Curryer kernel maker with config file and input dataframe

## Simulation Testing

In the event that we don't have real packets to start from (e.g. JPSS packets are not produced in ground testing)
we skip the entire L1A processing step and directly create a kernel from CSV simulated data from Ops.

Steps:

1. Read CSV data
2. Extract kernel data from CSV into pd.DataFrame for Curryer
3. Call Curryer kernel maker with config file and input dataframe
