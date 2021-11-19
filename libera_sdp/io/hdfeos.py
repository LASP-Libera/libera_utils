"""Module for HDF-EOS5 filehandling"""
# Standard
# Installed
import h5py as h5
import numpy as np
# Local


class HdfEos5(h5.File):
    """File object for HDF-EOS5 files

    Requirements and assertions made from: https://cdn.earthdata.nasa.gov/conduit/upload/4880/ESDS-RFC-008-v1.1.pdf
    """
    # TODO: Waiting to hear back from EOS on what is really required for these HDF objects

    def add_profile(self):
        pass

    def add_swath(self):
        pass

    def add_grid(self):
        pass

    def add_point(self):
        pass

    def add_zonal_average(self):
        pass
