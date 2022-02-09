"""Utils for HDF5 file handling"""
# Installed
import h5py as h5


def h5dump(f: h5.File or h5.Group, include_attrs: bool = True):
    """Prints the contents of an HDF5 object.

    Parameters
    ----------
    f: h5.File or h5.Group
        File, Group object from which to start inspecting.
    include_attrs: bool, Optional
        Default True.

    Returns
    -------
    None
    """
    def _print(name, obj):
        if isinstance(obj, h5.Group):
            print(f"Group:{name} ({len(obj)} members, {len(obj.attrs) if obj.attrs else 0} attributes)")
        elif isinstance(obj, h5.Dataset):
            print(f"Dataset:{name} "
                  f"(shape={obj.shape}, type={obj.dtype}, {len(obj.attrs) if obj.attrs else 0} attributes)")
        else:
            raise ValueError(f"Unrecognized object discovered in h5dump, of type {type(obj)}.")
        if include_attrs and obj.attrs:
            for key, val in obj.attrs.items():
                print(f"    @{key} = {val}")

    f.visititems(_print)
