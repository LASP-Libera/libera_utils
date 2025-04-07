# Working with NetCDF4 Files

NetCDF4 is a complete redesign of the NetCDF file format based on HDF5 data structures. i.e. all NetCDF4 files
are HDF5 files with some additional requirements and limitation of functionality.
Note that not all HDF5 files are NetCDF4 files. For more information on NetCDF5 and the underlying HDF5 structures,
see the documentation
[here](https://www.earthdata.nasa.gov/esdis/esco/standards-and-practices/netcdf-4hdf5-file-format).
There are several python packages (and libraries in other languages) that support reading and writing NetCDF4 files.
The SDC is using the Xarray python library.

The official documentation for Xarray is [here](https://docs.xarray.dev/en/stable/). It includes a much more comprehensive user guide with code examples.

Xarray builds on the numpy package, introducing labels for
multidimensional arrays in python. These labels come in the form of coordinates, dimensions, and attributes.
Xarray is broken into two main data structures: DataArrays and DataSets. DataArrays are contained within DataSets,
such that a single DataSet can hold multiple DataArrays. DataSets can then be written to NetCDF4 files.

## Reading NetCDF4 Files

To read NetCDF4 files we can use Xarray as well. NetCDF4 files have similar structure to HDF5 files.
NetCDF4 DataSets can have DataSets nested within one another. Here is an example of how to access each DataSet/Group.

```python
import xarray

with xarray.open_dataset("filename", group='/') as ds:
    print(ds) # print the highest level group
```

## Creating and Using DataArrays

[See documentation on `DataArray.to_netcdf` here.](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.to_netcdf.html)

DataArrays are arrays that can handle multiple dimensions with named or labeled axes. These DataArray objects add
metadata such as dimension names, coordinates, and attributes. DataArrays can be created from numpy arrays,
numpy-like arrays, pandas Series, and pandas DataFrames.

```python
import xarray as xr
import numpy as np
import pandas as pd

# Create the coordinates
time = pd.date_range(start='2022-01-01', periods=10, freq='D')  # 10 daily time steps
lat = np.linspace(-90, 90, 5)  # 5 latitude points from -90 to 90
lon = np.linspace(-180, 180, 8)  # 8 longitude points from -180 to 180

# Create random data
data = np.random.random((len(time), len(lat), len(lon)))

# Create the DataArray
data_array = xr.DataArray(data,
                          coords=[time, lat, lon],
                          dims=['time', 'lat', 'lon'])

# Display the DataArray
print(data_array)


print(data_array.values)  # the data in the object
print(data_array.dims)  # access the dimensions
print(data_array.coords)  # access the coors attribute
print(data_array.attrs)  # access metadata about the DataArray
```

You can create DataArrays across more dimensions as well. The number of variables in `dims` and `coords` should be
equal for multiple dimensions. You can also modify the DataArrays values with scalars.

```python
data_array.values = data_array.values * 2 # multiply the entire array by 2
```

## Writing DataSets as NetCDF4

[See documentation on `Dataset.to_netcdf` here.](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html)

Creating DataSets is similar to creating DataArrays, provide the data variables themselves, along with the coords,
dims and attributes you want to include. The data variables should be a dictionary with each key being the name of the
data and each value can be a DataSet, pandas dataframe or numpy array. Coords and Dims should be a dictionary as well.

For this example, we are creating the DataArrays first ,then will add them all into one DataSet and write that to a
NetCDF4 file. When creating DataSets, Xarray will often infer the dims from the coords and data variables given, if
you do not pass any while creating DataSets. When writing to the NetCDF4 files if different variables “lie” on
different dimensions, they will smash them together and replace the extra values (when viewed in a file viewer)
with zeros. When writing a file using Xarray, using the engine “h5netcdf” will write the file faster.


```python
import random
import pandas
import numpy
import xarray as xr

data_length = random.randint(1000,2000) # creating random vector length to simulate data

# creating multiple data fields to simulate fake data
times = pandas.date_range("2014-09-06", periods=data_length)
short_wave = numpy.random.rand(data_length)
long_wave = numpy.random.rand(data_length)
total_radiance = numpy.random.rand(data_length)
split_radiance = numpy.random.rand(data_length)

# creating the DataSets from the created fields
short_wave = xr.DataArray(short_wave, coords=[times], dims=['times'])
long_wave = xr.DataArray(long_wave, coords=[times], dims=['times'])
total_rad = xr.DataArray(total_radiance, coords=[times], dims=['times'])
split_rad = xr.DataArray(split_radiance, coords=[times], dims=['times'])

# create the DataSet
ds = xr.Dataset({
    'short_wave': short_wave,
    'long_wave': long_wave,
    'total_radiance': total_rad,
    'split_rad': split_rad
},
    coords={'times': times},
)

# Write some metadata
ds.attrs["ALGORITHM_VERSION"] = "3.14.159"

# write to a NetCDF4 file
ds.to_netcdf('filename', group="/", mode='a', engine='h5netcdf')
```

You can specify group structure with group keyword, similar to a filesystem path (/groups/are/paths). When writing
multiple DataSets to a file or if you need to append them, use keyword “mode” with value “a” to append.
