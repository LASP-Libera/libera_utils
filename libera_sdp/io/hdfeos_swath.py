"""First pass at module for Swath HDF-EOS5 filehandling"""
# Standard
import h5py as h5
import numpy as np
from datetime import date


class SwathHdfEos5(h5.File):
    """Creates structure for hdf5 swath file requirements.
    Note: Requirements and assertions from: https://cdn.earthdata.nasa.gov/conduit/upload/4880/ESDS-RFC-008-v1.1.pdf
    TODO: Revisit with more up to date requirements in future and add necessary exceptions
    """

    def __init__(self, path, filename, file_version, doi):
        """Create a hdf5 file

        Parameters
        ----------
        path : str
            File directory
        filename : str
            Filename
        file_version : str in the form of #.#.#
            File version
        doi : str
            doi information
        """
        self.__dir = '/'.join([path, filename])
        self.__hierarchicalFile = h5.File(self.__dir, "w")
        self.__file_version = file_version
        self.__doi = doi

    def create_swath_groups(self, swath_names):
        """Create swath groups and subgroups

        Parameters
        ----------
        swath_names : list of str
            List of swath names
        """
        grp1 = self.__hierarchicalFile.create_group('HDFEOS')
        grp3 = grp1.create_group("SWATHS")
        grp4 = grp1.create_group("ADDITIONAL")
        grp4.create_group('FILE_ATTRIBUTES')

        grp2 = self.__hierarchicalFile.create_group('HDFEOS INFORMATION')
        grp2.attrs['HDFEOSVersion'] = '_'.join(['HDFEOS', self.__file_version])

        for i in swath_names:
            grp5 = grp3.create_group(i)

            grp5.create_group('DataField')
            grp5.create_group('GeoField')
            grp5.create_group('ProfileField')
            grp5.create_group('Dimension')

    def add_swath_dataset(self, dataset_path, dataset_names, datasets, dataset_units):
        """Create datasets in directory defined by dataset path.

        Parameters
        ----------
        dataset_path : str
            Location of dataset
        dataset_names : list of str
            Name of datasets
        datasets : list of numpy arrays
            Actual datasets
        dataset_units : list of str
            Dataset units
        """
        with h5.File(self.__dir, mode='a') as f:

            for i in range(len(dataset_names)):

                d1 = f[dataset_path].create_dataset(
                    dataset_names[i], datasets[i].shape, data=datasets[i], dtype=datasets[i].dtype)

                d1.attrs['_FillValue'] = -9999.0
                d1.attrs['units'] = dataset_units[i]

    def add_swath_file_attr(self):
        """Add file attributes.
        Note: this was modeled after a template sent by sdps-support@earthdata.nasa.gov :
        AMSR_E_L2_Rain_V13_200706062353_D.he5
        """
        with h5.File(self.__dir, mode='a') as f:

            f.attrs['institution'] = "NASA's EVC1 Science Investigator-led Processing System"
            f.attrs['references'] = ''.join(['Please cite these data as: Peter Pilewskie. ', str(date.today().year),
                                             '. Libera L2 Data Product, Version ', str(self.__file_version),
                                             '. Laboratory for Atmospheric and Space Physics. Boulder, Colorado, USA.',
                                             ' doi: ', str(self.__doi)])
            f.attrs['source'] = 'satellite observation'
            f.attrs['title'] = 'Level 2 Libera Data'

    def descend_obj(self, obj, swaths, string, i=0, j=0, group=None):
        """
        Iterates through groups in a HDF5 file and creates a string of the groups and objects

        Parameters
        ----------
        obj : h5py._hl.group.Group
            Initial group
        swaths : numpy array of strings
            Swath names
        string : str
            Passes updated string information
        i : int
            Initial value for swath numbering
        j : int
            Initial value for object numbering
        group : str
            Passes group information

        Returns
        ----------
        string : Metadata string
        """

        if type(obj) is h5._hl.group.Group:

            for key in obj.keys():
                if key in swaths:
                    i += 1
                    swath_group = '_'.join(['GROUP=SWATH', str(i)])
                    swath_name = ''.join(['SwathName=', key])

                    string = ''.join([string, '\n\t', swath_group, '\n\t\t', swath_name])
                    string = self.descend_obj(obj[key], swaths, string, i=i)
                    string = ''.join([string, '\n\t', 'END_', swath_group])
                elif type(obj[key]) is h5._hl.group.Group:
                    string = ''.join([string, '\n\t\t', 'GROUP=', key])
                    string = self.descend_obj(obj[key], swaths, string, group=key)
                    string = ''.join([string, '\n\t\t', 'END_GROUP=', key])
                else:
                    j += 1
                    tri_indent = '\n\t\t\t'
                    object_number = '_'.join([group, str(j)])
                    object_string = ''.join([tri_indent, 'OBJECT=', object_number])
                    name = ''.join([tri_indent, '\t', group, 'Name= ', key])
                    datatype = ''.join([tri_indent, '\t', 'Datatype= ', str(obj[key].dtype)])
                    end_object = ''.join([tri_indent, 'END_OBJECT=', object_number])

                    string = ''.join([string, object_string, name, datatype, end_object])
                    string = self.descend_obj(obj[key], swaths, string, j=j)

        return string

    def add_swath_metadata(self):
        """
        Add StructMetadata to HDFEOS INFORMATION

        """
        swaths = np.array([])

        with h5.File(self.__dir, 'r') as f:
            string = 'GROUP=SwathStructure'
            obj = f['HDFEOS/SWATHS']

            for key in obj.keys():
                swaths = np.append(swaths, key)

            string = self.descend_obj(obj, swaths, string)
            f['HDFEOS INFORMATION'].create_dataset('StructMetadata.0', (1,), data=string)

        return
