# Working with NetCDF4 Files

Note: This example uses Xarray, but we have not fully committed to using the Xarray library yet. 

NetCDF4 is a complete redesign of the NetCDF file format based on HDF5 data structures. i.e. all NetCDF4 files 
are HDF5 files with some additional requirements and limitation of functionality. 
Note that not all HDF5 files are NetCDF4 files. For more information on NetCDF5 and the underlying HDF5 structures,
see the documentation 
[here](https://www.earthdata.nasa.gov/esdis/esco/standards-and-practices/netcdf-4hdf5-file-format).
There are several Python packages (and libraries in other languages) that support reading and writing NetCDF4 files. 
The SDC is using the Xarray python library. 

[Documentation on Xarray here.](https://docs.xarray.dev/en/stable/)

Xarray introduces labels for 
multidimensional arrays in python. These labels come in the form of coordinates, dimensions, and attributes. 
Xarray is broken into two main data structures: DataArrays and Datasets. DataArrays are contained within Datasets, 
so you can create multiple DataArrays and compose them into a single Dataset. DataSets can then be written to 
NetCDF4 files using one of several underlying HDF5 file drivers.

## Reading NetCDF4 Files

To read netCDF4 files we can use Xarray as well. NetCDF4 files have similar structure to HDF5 files. NetCDF4 DataSets can have DataSets nested within one another. Here is an example of how to access each DataSet/Group.

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
import xarray 
import numpy
import pandas 
data = numpy.random.rand(4,3)  # creating 4x3 matrix/array of random numbers

locations = ["IA" ,"IL" ,"IN"]  # random locations

times = pandas.date_range("200-01-01", periods=4)  # creating random times (you don't need pandas generally)

da = xarray.DataArray(
	data,
	coords = [times, locations],
	dims = ["time", "space"]
)

print(da.values)  # the data in the object
print(da.dims)  # access the dimensions 
print(da.coords)  # access the coors attribute
print(da.attrs)  # access metadata about the DataArray
```

You can create DataArrays across more dimensions as well. The number of variables in `dims` and `coords` should be 
equal for multiple dimensions. You can also modify the DataArrays values with scalars.

```python
da.values = da.values * 2 # multiply the entire array by 2
```

## Writing DataSets as NetCDF4

[See documentation on `Dataset.to_netcdf` here.](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html)

Creating DataSets is similar to creating DataArrays, provide the data variables themselves, along with the coords, 
dims and attributes you want to include. The data variables should be a dictionary with each key being the name of the 
data and each value can be a DataSet, pandas dataframe or numpy array. Coords and Dims should be a dictionary as well. 

For this example, we are creating the DataArrays first ,then will add them all into one DataSet and write that to a 
NetCDF4 file. When creating DataSets, xarray will often infer the dims from the coords and data variables given, if 
you do not pass any while creating DataSets. When writing to the netCDF4 files if different variables “lie” on 
different dimensions, they will smash them together and replace the extra values (when viewed in a file viewer) 
with zeros. When writing a file using Xarray, using the engine “h5netcdf” will write the file faster.


```python
import random
import pandas 
import numpy 
import xarray as xr

vector_length = random.randint(1000,2000) # creating random vector length to simulate data

# creating multiple vectors to simulate fake data
times = pandas.date_range("2014-09-06", periods=vector_length)
short_wave = numpy.random.rand(vector_length)
long_wave = numpy.random.rand(vector_length)
total_radiance = numpy.random.rand(vector_length)
split_radiance = numpy.random.rand(vector_length)

# creating the DataSets from the created vectors
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

# write to a netcdf4 file
ds.to_netcdf('filename', group="/", mode='a', engine='h5netcdf')
```

You can specify group structure with group keyword, similar to a filesystem path (/groups/are/paths). When writing 
multiple Datasets to a file or if you need to append them, use keyword “mode” with value “a” to append.
