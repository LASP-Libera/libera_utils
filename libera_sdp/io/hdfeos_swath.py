"""Module for Swath HDF-EOS5 filehandling"""
# Standard
import h5py as h5
import json
import numpy as np
from datetime import date


class SwathHdfEos5(h5.File):
    """Creates structure for hdf5 swath file requirements.
    Note: Requirements and assertions from: https://cdn.earthdata.nasa.gov/conduit/upload/4880/ESDS-RFC-008-v1.1.pdf
    """

    def __init__(self, path: str, attribute_path: str, **kwargs):
        """Initialize upstream attributes and modifies path in attributes.json

        Parameters
        ----------
        path : str
            Path for hdf5 file.
        attribute_path : str
            Path for json file attributes.
        """

        attributes = open(attribute_path, "r")
        json_object = json.load(attributes)
        attributes.close()

        json_object["path"] = path

        attributes = open(attribute_path, "w")
        json.dump(json_object, attributes, indent=4)
        attributes.close()

        attributes = open(attribute_path, "r")
        self.attributes = json.load(attributes)

        super().__init__(path, **kwargs)

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
            self, dataset_path: str, dataset_names: list, datasets: list, dataset_units: list, fill_value=-9999.0):
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
        self.attrs['references'] = ''.join(['Please cite these data as: ', self.attributes['pi'], '. ',
                                            str(date.today().year), '. ', self.attributes['title'],
                                            ', Version ', self.attributes['version'],
                                            '. ', self.attributes['institute'], '.', self.attributes['doi']])
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
    def validate(cls, thisdict):
        """
        Class method for validation

        Parameters
        ----------
        thisdict : dict
            Dictionary containing information for hdf5 file.

        """
        val = cls(thisdict['path'], thisdict['attribute_path'], mode="w")
        val.add_swath_file_attr()
        val.create_swath_groups(thisdict['swath_names'])
        val.add_swath_dataset(
            thisdict['dataset_path'], thisdict['dataset_names'], thisdict['datasets'], thisdict['dataset_units'])
        val.add_swath_metadata()

        val.validate_self(thisdict)

        return

    def validate_self(self, thisdict):
        """Validates self.

        Parameters
        ----------
        thisdict : dict
            Dictionary containing information for hdf5 file.

        """

        for i in thisdict['swath_names']:
            assert(i in self['HDFEOS/SWATHS'].keys()), f"{i} is missing"

        for i in thisdict['dataset_names']:
            assert(i in self[thisdict['dataset_path']].keys()), \
                '/'.join([f"{i} is missing from", thisdict['dataset_path'][0]])

        for i in range(len(thisdict['dataset_names'])):
            unit_path = '/'.join([thisdict['dataset_path'], thisdict['dataset_names'][i]])
            assert(self[unit_path].attrs['units'] == thisdict['dataset_units'][i])
