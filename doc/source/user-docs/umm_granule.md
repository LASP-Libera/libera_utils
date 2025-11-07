# What is UMM Granule?

UMM stands for Unified Metadata Model, which is a standardized metadata format used by NASA to describe Earth science data products. [UMM Granule](<https://wiki.esipfed.org/Data_Discovery_(UMM-Granule)>) specifically refers to the metadata associated with individual data granules, which are the smallest units of data that can be independently accessed and processed.

# How do I create a UMM Granule definition for my dataset?

If you have an xarray Dataset, which can be created from NetCDF files using the xarray.load_dataset function, you can use the `libera_utils` library to generate a UMM Granule definition. The library provides tools to extract relevant metadata from the Dataset and format it according to the UMM Granule standard.
Here is an example of how to create a UMM Granule definition from an xarray Dataset:

```python
import xarray as xr
from libera_utils.umm_granule import create_umm_granule_from_xarray
# Load your dataset (replace 'your_dataset.nc' with your actual file)
ds = xr.load_dataset('your_dataset.nc')
# Create UMM Granule definition
umm_granule = create_umm_granule_from_xarray(ds)
# Print or save the UMM Granule definition
print(umm_granule)
```

The `UMMGranule` class creates a model that adheres to the UMM Granule specification.
