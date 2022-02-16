"""First pass at module for Swath HDF-EOS5 filehandling"""
# Standard
import h5py as h5
import json
import numpy as np
from datetime import date


class SwathHdfEos5(h5.File):
    """Creates structure for hdf5 swath file requirements.
    Note: Requirements and assertions from: https://cdn.earthdata.nasa.gov/conduit/upload/4880/ESDS-RFC-008-v1.1.pdf

    Attributes:
        file_version : str in the form of #.#.#
            File version.
        doi : str
            doi information
    TODO: Revisit with more up to date requirements in future and add necessary exceptions
    """

    path = '../../libera_sdp/data/hdf5'
    filename = 'swath_test.he5'

    attribute_path = '../../libera_sdp/data/hdf5'
    attribute_filename = 'attributes.json'


    def __init__(self, path: str, filename: str, **kwargs):
        """Create a hdf5 file

        Parameters
        ----------
        path : str
            File directory
        filename : str
            Filename
        """
        self.dir = '/'.join([path, filename])
        self.config = '/'.join([self.attribute_path, self.attribute_filename])

        #load attributes
        self.attributes = open(self.config, "r")
        json_object = json.load(self.attributes)
        self.attributes.close()

        json_object["path"] = path
        json_object["filename"] = filename

        self.attributes = open(self.config, "w")
        json.dump(json_object, self.attributes, indent=4)
        self.attributes.close()

        super().__init__(self.dir, **kwargs)


    def create_swath_groups(self, swath_names: list):
        """Create swath groups and subgroups

        Parameters
        ----------
        swath_names : list of str
            List of swath names
        """
        grp1 = self.create_group('HDFEOS')
        grp3 = grp1.create_group("SWATHS")
        grp4 = grp1.create_group("ADDITIONAL")
        grp4.create_group('FILE_ATTRIBUTES')

        grp2 = self.create_group('HDFEOS INFORMATION')
        grp2.attrs['HDFEOSVersion'] = '_'.join(['HDFEOS', self.attributes['version']])

        for i in swath_names:
            grp5 = grp3.create_group(i)

            grp5.create_group('DataField')
            grp5.create_group('GeoField')
            grp5.create_group('ProfileField')
            grp5.create_group('Dimension')

    def add_swath_dataset(
            self, dataset_path: str, dataset_names: list, datasets: list, dataset_units: list, fill_value = -9999.0):
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
        fill_value : float (optional)
            Fill value
        """

        for i in range(len(dataset_names)):

            d1 = self[dataset_path].create_dataset(
                dataset_names[i], datasets[i].shape, data=datasets[i], dtype=datasets[i].dtype)

            d1.attrs['_FillValue'] = fill_value
            d1.attrs['units'] = dataset_units[i]

    def add_swath_file_attr(self):
        """Add file attributes.
        Note: this was modeled after a template sent by sdps-support@earthdata.nasa.gov :
        AMSR_E_L2_Rain_V13_200706062353_D.he5
        """

        self.attrs['institution'] = self.attributes['institution']
        self.attrs['references'] = ''.join(['Please cite these data as: ', self.attributes['pi'],'. ', str(date.today().year),
                                             '. ', self.attributes['title'],', Version ', self.attributes['version'],
                                             '. ', self.attributes['institute'],'.',
                                             self.attributes['doi']])
        self.attrs['source'] = self.attributes['source']
        self.attrs['title'] = self.attributes['title']

    @classmethod
    def create_metadata_struct(cls, obj, swaths, string, i=0, j=0, group=None):
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

        if isinstance(obj, h5.Group):

            for key in obj.keys():
                if key in swaths:
                    i += 1
                    swath_group = '_'.join(['GROUP=SWATH', str(i)])
                    swath_name = ''.join(['SwathName=', key])

                    string = ''.join([string, '\n\t', swath_group, '\n\t\t', swath_name])
                    string = cls.create_metadata_struct(obj[key], swaths, string, i=i)
                    string = ''.join([string, '\n\t', 'END_', swath_group])
                elif isinstance(obj[key], h5.Group):
                    string = ''.join([string, '\n\t\t', 'GROUP=', key])
                    string = cls.create_metadata_struct(obj[key], swaths, string, group=key)
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
                    string = cls.create_metadata_struct(obj[key], swaths, string, j=j)

        return string

    def add_swath_metadata(self):
        """
        Add StructMetadata to HDFEOS INFORMATION

        """
        swaths = np.array([])

        string = 'GROUP=SwathStructure'
        obj = self['HDFEOS/SWATHS']

        for key in obj.keys():
            swaths = np.append(swaths, key)

        string = self.create_metadata_struct(obj, swaths, string)
        self['HDFEOS INFORMATION'].create_dataset('StructMetadata.0', (1,), data=string)

        return

    @classmethod
    def validate(cls, source):
        """
        Class method for validation

        """

        #cls.add_swath_file_attr()
        #cls.create_swath_groups(swath_names)
        #cls.add_swath_dataset(dataset_path, dataset_names, datasets, dataset_units)
        #cls.add_swath_metadata()
        print('hi')

if __name__ == "__main__":

    filename = 'swath_test.he6'
    path = '../../libera_sdp/data/hdf6'

    swath_names = ['Swath1']
    dataset_path = 'HDFEOS/SWATHS/Swath1/DataField'
    dataset_names = ['Temperature', 'SunglintAngle']
    datasets = [np.array([1, 1]), np.array([2, 2])]
    dataset_units = ['Kelvin', 'radians']

    validate_instance = SwathHdfEos5(path, filename, mode="w")
    SwathHdfEos5.add_swath_file_attr(validate_instance)
    SwathHdfEos5.create_swath_groups(validate_instance,swath_names)
    SwathHdfEos5.add_swath_dataset(validate_instance, dataset_path, dataset_names, datasets, dataset_units)
    SwathHdfEos5.add_swath_metadata(validate_instance)

    print('hi')











