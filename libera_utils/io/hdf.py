"""Utils for HDF5 file handling"""
# Installed
import h5py as h5


def h5dump(f: h5.File or h5.Group, include_attrs: bool = True, stdout: bool = False):
    """Prints the contents of an HDF5 object.

    Parameters
    ----------
    f: h5py.File or h5py.Group
        File, Group object from which to start inspecting.
    include_attrs: bool, Optional
        Default True.
    stdout: bool, Optional
        Default False. If True, prints to stdout as the object tree is traversed.

    Returns
    -------
    : str
        Concatenated string of HDF5 contents
    """
    if not isinstance(f, (h5.File, h5.Group)):
        raise ValueError(f"Invalid input to h5dump. H5 object f must be a File or Group. Got {type(f)}")
    srep = []

    def _print(name, obj):
        if isinstance(obj, h5.Group):
            s = f"Group:{obj.name} ({len(obj)} members, {len(obj.attrs) if obj.attrs else 0} attributes)"
            srep.append(s + '\n')
        elif isinstance(obj, h5.Dataset):
            s = (f"Dataset:{obj.name} "
                 f"(shape={obj.shape}, type={obj.dtype}, {len(obj.attrs) if obj.attrs else 0} attributes)")
            srep.append(s + '\n')
        elif isinstance(obj, h5.Datatype):
            s = f"Datatype:{obj.name} {obj}"
            srep.append(s + '\n')
        else:
            raise ValueError(f"Unrecognized object discovered in h5dump, of type {type(obj)}.")

        if stdout:
            print(s)

        if include_attrs and obj.attrs:
            for key, val in obj.attrs.items():
                s = f"    @ {key} = {val}"
                srep.append(s + '\n')
                if stdout:
                    print(s)

    top_obj_name = f.name if f.name == '/' else f.name[1:]  # Creates a name similar to that passed by visititems
    _print(top_obj_name, f)
    f.visititems(_print)
    return "".join(srep)
