# Creating Libera Data Product Files (NetCDF-4)

The Libera SDC provides utilities for creating NetCDF4 data product files that conform to YAML data product definitions
and a few global expectations. These definitions ensure a degree of consistency across all Libera data products,
including proper metadata, coordinate systems, and data types. Libera Utils includes a few different ways to work with
data product definitions for validation.

## Product Definition YAML Structure

Product definition files specify the exact structure and metadata for data products. For example:

```yaml
# Product level metadata attributes
attributes:
  ProductID: RAD-4CH
  version: 0.0.1

# Coordinate variables
# Coordinates with the same name as their dimension are "dimension coordinates" and are
# automatically linked to variables referencing their dimension name.
coordinates:
  radiometer_time:
    dtype: datetime64[ns]
    dimensions: ["radiometer_time"]
    attributes:
      long_name: "Time of sample collection"
    encoding:
      units: nanoseconds since 1958-01-01
      calendar: standard
      dtype: int64

# Data variables
variables:
  fil_rad:
    dtype: float64
    dimensions: ["radiometer_time"]
    attributes:
      long_name: Filtered Radiance
      units: W/(m^2*sr*nm)
      valid_range: [0, 1000]
    encoding:
      chunksizes: [1024]
```

The `LiberaDataProductDefinition` system ensures all Libera data products maintain consistent structure, metadata, and
quality standards across the mission.

### Dimensions

All dimensions referenced in a product definition must match an existing dimension in the global allowed dimension list
in `libera_utils/data/libera_dimensions.yml`. Additionally, data referencing a dimension must have the correct size if
referencing a dimension with a static size. Most dimensions are considered to be "dynamic size", in which case the sizes
of all variables/coordinates referencing the dimension must match but no specific size is enforced.

### Attributes

Attributes may be defined at the product (Dataset) level or the variable (DataArray) level. All products must contain a
set of expected product level attributes, defined in `libera_utils/data/required_product_attributes.yml` to be valid
(these will be added automatically when using `write_libera_data_product`). In addition, attributes may be defined in a
product definition itself.

Attributes defined as `null` or empty are considered "required dynamic attributes". Their values must be set before the
product Dataset is considered valid but the value is expected to be dynamic. To include dynamic attribute values when
writing a data product, pass the `dynamic_product_attributes` kwarg to `write_libera_data_product`.

The precedence for attribute assignment is: `required_product_attributes.yml` (globally required) <
`product_definition_file.yml` (defined in product definition) < `dynamic_product_attributes` kwarg. For example,
`ProductID` is required by the standard metadata file but it is dynamic (`null` valued). It could be defined with a
value in a particular product definition yml file or it could be passed via kwarg to `write_libera_data_product`.

### Chunking

NetCDF4 stores each variable in fixed-size blocks called chunks, and reads, writes and compresses
a whole chunk at a time. A chunk shape that matches how the product is read, such as one camera
frame (`[1, 512, 512]`) for a product read frame by frame, keeps a reader from pulling in data it
does not need. A chunk shape that matches the Dask chunks the product is computed in lets each
worker write whole chunks. Without `chunksizes`, the engine picks a shape itself.

Give `encoding.chunksizes` as a list of positive integers, one per entry in `dimensions`, in
dimension order. Anything else, including a count that does not match, raises `ValidationError`
when the definition is loaded, so a bad chunk shape fails before any data is processed.

### Compressing chunks on Dask workers

A product computed from Dask arrays under the `dask.distributed` scheduler is otherwise
compressed one chunk at a time: every block reaches HDF5 through a single write lock, and the
shuffle and deflate filters run inside it. For a large product that serial compression can be
most of the run. Passing `worker_compressed_variables`, a tuple of Dask-backed data variable
names, has the workers shuffle and deflate each of those variables' chunks in parallel. The
writing process then only stores the compressed bytes with h5py `write_direct_chunk`.

```python
write_libera_data_product(
    data_product_definition=product_definition_path,
    data=dataset,
    output_path=output_dir,
    time_variable="CAMERA_TIME",
    worker_compressed_variables=("Radiance", "Latitude", "Longitude"),
)
```

The keyword is optional; without it the product is written with `to_netcdf` as before. List the
large Dask-backed variables whose compression dominates the write. Dask-backed variables left off
the list are still written, compressed by the writing process one chunk at a time.

The file reads back the same as a normal write, with the same metadata, chunk shapes, filters
and stored chunks. Every Dask block is computed in one graph, so a task that feeds several
variables runs once.

It requires the `h5netcdf` engine and an active `distributed.Client`. Each named variable must
be chunked on disk with deflate, optionally after shuffle, have an integer or floating-point
dtype, and have Dask chunk lengths that are whole multiples of its on-disk chunk lengths. Any
other Dask-backed variable must be numeric or boolean. Anything else raises `ValueError` before
any data is computed. A computed block whose dtype or shape differs from what its Dask array
declared raises `ValueError`, naming the variable, when it arrives. Once the write has created
its local file, any error deletes that file before propagating. For a local destination, a
product already at that path is gone as well; for an S3 destination, the existing object is
untouched and nothing is uploaded.

## Basic Usage

The simplest way to create a valid Libera data product is using the `write_libera_data_product` function:

```python
import numpy as np
from pathlib import Path
from libera_utils.io.netcdf import write_libera_data_product

# Define the path to the data product definition YAML file
product_definition_path = Path("path/to/product_definition.yml")

# Create numpy data arrays for each variable and coordinate required
n_times = 100
time_data = np.arange(n_times).astype("datetime64[ns]") + np.datetime64("2024-01-01")
lat_data = np.linspace(-90, 90, n_times)
lon_data = np.linspace(-180, 180, n_times)
fil_rad_data = np.random.rand(n_times) * 1000  # Filtered radiance values
q_flag_data = np.zeros(n_times, dtype=np.int32)  # Quality flags

# Combine all data into a dictionary
data = {
    "time": time_data,
    "lat": lat_data,
    "lon": lon_data,
    "fil_rad": fil_rad_data,
    "q_flag": q_flag_data,
}

# Call write_libera_data_product with correct arguments
output_dir = Path("/path/to/output/directory")
filename = write_libera_data_product(
    data_product_definition=product_definition_path,
    data=data,
    output_path=output_dir,
    time_variable="radiometer_time",  # Specify which variable contains time data
    dynamic_product_attributes={"algorithm_version": "1.2.3"},
    strict=True,  # Enforce strict conformance, raising exceptions on errors
)

print(f"Created data product: {filename.path}")
# Output: Created data product: /path/to/output/directory/LIBERA_L1B_RAD-4CH_V0-0-1_20240101T000000_20240101T002739_R20270103T080910123456.nc
```

The function automatically:

- Validates data against the product definition
- Sets the `date_created` attribute to the creation time, overwriting (with a warning) any value already set
- Generates a standardized filename from the product, algorithm version, and time range, whose revision is that
  `date_created`
- Applies correct encoding and compression settings
- Ensures CF-convention compliance

## Advanced Usage

There are a few other APIs that may be useful for debugging data product creation during development. Once a data
product definition is working, the proper approach is to call `write_libera_data_product` with `strict=True` (the
default).

### Creation, Enforcement, and Validation for Existing Dataset

You can use the `LiberaDataProductDefinition` class to create, enforce (modify), and validate xarray Dataset objects.

#### Validation of Conformance

The `LiberaDataProductDefinition.check_dataset_conformance()` method takes a dataset as an argument and checks that it
conforms to the data product definition. The typical usage of this method is in strict mode, which raises an exception
for any validation error. For debugging, you can run with `strict=False`, in which case it returns a (possibly empty)
list of informational error messages.

`date_created` is a required attribute. `create_product_dataset` sets it, and `write_libera_data_product` sets it on a
Dataset passed in directly. A conformance check on a Dataset built some other way reports `date_created` as missing
until the writer sets it; that is expected, and setting it yourself only makes the writer overwrite it with a warning.

```python
# Call LiberaDataProductDefinition.check_dataset_conformance() with strict=False
# so we get a list of error messages instead of an exception raised.
# This checks the conformance of an existing dataset against a data product definition without modifying it
# Note that the dataset need not be generated by create_product_dataset. This method accepts _any_ xarray Dataset object.
validation_errors = dpd.check_dataset_conformance(dataset, strict=False)

# Demonstrate how to use the return values
if validation_errors:
    print(f"\nValidation found {len(validation_errors)} issues")
    print("Attempting to fix issues with enforce_dataset_conformance...")
```

#### Enforcement of Product Definition

The `LiberaDataProductDefinition.enforce_dataset_conformance()` method takes a Dataset as an argument and attempts to
enforce all data product definition requirements by modifying any necessary attributes, data types, and encodings.
Fixable issues (wrong attribute values, encoding mismatches, safe dtype casts) are corrected automatically and
reported via `WARNING` or `DEBUG` log messages. Issues that cannot be automatically fixed (e.g., dimension
mismatches, unsafe dtype casts) raise a `ValueError`.

```python
# Call dpd.enforce_dataset_conformance() to fix problems with a dataset
# This modifies the dataset in-place to fix what it can.
# Unfixable issues (e.g., dimension mismatches, unsafe dtype casts) raise ValueError.
fixed_dataset = dpd.enforce_dataset_conformance(dataset)

# Save the fixed dataset
fixed_dataset.to_netcdf("output_fixed.nc")
```

#### Direct Creation of Conforming Dataset

```python
import numpy as np
import xarray as xr
from pathlib import Path
from libera_utils.io.product_definition import LiberaDataProductDefinition

# Create LiberaDataProductDefinition object from a YAML file
definition_path = Path("path/to/product_definition.yml")
dpd = LiberaDataProductDefinition.from_yaml(definition_path)

# Prepare your data as numpy arrays
n_times = 50
data = {
    "time": np.arange(n_times).astype("datetime64[ns]") + np.datetime64("2024-03-01"),
    "lat": np.linspace(-90, 90, n_times),
    "lon": np.linspace(-180, 180, n_times),
    "fil_rad": np.random.rand(n_times) * 500,
    "q_flag": np.random.randint(0, 100, n_times, dtype=np.uint32),
}

# Create a Dataset with dpd.create_product_dataset().
# This method constructs the Dataset from numpy arrays, assigning each variable to
# coordinates or data variables as defined in the product definition, and applying
# the static attributes and encodings from the definition.
# It does NOT enforce conformance or coerce dtypes — use enforce_dataset_conformance() for that.
dataset = dpd.create_product_dataset(data=data)
```

### Dynamic Global and Variable Attributes

_Note: L2 Developers should not need this._

Avoid this unless absolutely necessary!

When creating datasets, you can specify attributes whose values are not defined in the YAML (i.e. they are null). We
call these "dynamic attributes" and they are required but must be passed directly during dataset creation:

```python
# Set dynamic global attribute values (e.g., algorithm-specific metadata values for required attribute keys)
dynamic_product_attrs = {
    "algorithm_version": "2.1.0",
    "processing_level": "L1B",
}

# Set dynamic variable attribute values
# NOTE: these should be avoided unless absolutely necessary
dynamic_var_attrs = {
    "fil_rad": {
        "calibration_date": "2024-01-01",
        "sensor_id": "LIBERA-001",
    },
    "q_flag": {
        "flag_meanings": "good questionable bad",
        "flag_values": [0, 1, 2],
    },
}

# Create dataset with custom attributes
dataset = dpd.create_product_dataset(
    data=data,
    dynamic_product_attributes=dynamic_product_attrs,
    dynamic_variable_attributes=dynamic_var_attrs,
)
```
