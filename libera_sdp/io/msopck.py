"""Python interface to msopck, make text data into a CK file"""
import os
import subprocess


_msopck_defaults={
    "INPUT_DATA_TYPE": "MATRICES",
    "INPUT_TIME_TYPE": "ET",
    "ANGULAR_RATE_PRESENT": 'MAKE UP/NO AVERAGING',
    "CK_TYPE": 3
}


def print_value(kwarg,value,ouf):
    """
    Print a value to an MSOPCK setup file

    :param kwarg: Keyword to use
    :param value: Value. May be string, scalar integer or floating-point, or 1D array integer or floating-point
    :param ouf: File-like object to write to
    :return: None, but as a side-effect, writes the parameter to the given setup file
    """
    if isinstance(value, str):
        print("%s = '%s'" % (kwarg, value), file=ouf)
    else:
        try:
            s="%s = ("%kwarg
            for i,v in enumerate(value):
                if i!=0:
                    s=s+","
                if isinstance(v,str):
                    s=s+"'"+v+"'"
                elif isinstance(v, int):
                    s=s+"%d"%v
                else:
                    s=s+"%.16f"%v
            s=s+")"
            print(s, file=ouf)
        except:
            if isinstance(value, int):
                print("%s = %d" % (kwarg, value), file=ouf)
            else:
                print("%s = %.16f" % (kwarg, value), file=ouf)


def msopck(t,data,av=None,oufn=None,new_kernel=True,keep_temp=True,msopck_oufn=None,data_oufn=None,temp_suffix="",comment="",**kwargs):
    """
    Use the MSOPCK utility to make a CK file from a table of pointing data

    :param t: 1D iterable of ephemeris time stamps
    :param data: 2D array of pointing data. First index is time, second index is
                 vector/matrix/quaternion/euler angle component. Size of first dimension
                 must equal that of t, and size of second dimension must be suitable
                 for the kind of pointing you are using, IE 4 for quaternion, 9 for matrix, 3 for Euler angles
    :param av: 2D array of angular velocity data. First index is time, second is angular
               velocity component. Size of second dimension must be 3.
    :param oufn: Filename of kernel to write to
    :param new_kernel: True if this should be made a new kernel. False will add a segment to
                       an existing kernel (if it already exists)
    :param keep_temp: If true, keep temporary files (parameter and data files), otherwise
                      delete them when done
    :param msopck_oufn: If set, use this as the name of the parameter file for msopck. Default
                      calculates automatically from oufn
    :param data_oufn: If set, use this as the name of the data file. Default calculates
                      automatically from oufn
    :param temp_suffix: If set, use this when calculating the filenames for msopck_oufn and data_oufn
    :param comment: If set, include this string in the comment in the CK file. May be a multi-line string.
    :param kwargs: Each extra named keyword argument gets passed as a parameter to MSOPCK, by being written
                   in the parameter file with the given name. For example, if you want to include a line

                   FOO='BAR'

                   in the parameter file for MSOPCK, call this function like this:

                   msopck(...,FOO='BAR')

                   I don't think that Spice is case-sensitive, but the parameter written to the file will
                   follow the case of the keyword parameter in Python.

                   The exact parameters you want to use and their usages are documented in the MSOPCK
                   user guide and the CK required reading. It's almost impossible to use this function
                   without understanding those two references.
    :return: None, but has side effect of CK file pointed to by oufn now exists

    """
    if new_kernel:
        try:
            os.unlink(oufn)
        except:
            #swallow errors, since deleting a nonexistent file throws an error and all other errors will happen below as well
            pass
    if msopck_oufn is None:
        msopck_oufn=oufn+temp_suffix+".msopck"
    if data_oufn is None:
        data_oufn=oufn+temp_suffix+".data"
    with open(msopck_oufn,"wt") as msopck_ouf:
        try:
            for line in comment:
                print(line,file=msopck_ouf)
        except:
            print(comment,file=msopck_ouf)
        print("\\begindata",file=msopck_ouf)
        for kwarg,value in kwargs.items():
            print_value(kwarg,value,msopck_ouf)
        for kwarg,value in _msopck_defaults.items():
            if kwarg not in kwargs:
                print_value(kwarg, value, msopck_ouf)

    with open(data_oufn,"wt") as data_ouf:
        for i in range(t.size):
            print("%20.9f"%t[i],end='',file=data_ouf)
            for val in data[i,:]:
                print("  %24.16e"%val,end='',file=data_ouf)
            if av is not None:
                for val in av[i,:]:
                    print("  %24.16e"%val,end='',file=data_ouf)
            print(file=data_ouf)

    old=os.getcwd()
    os.chdir(os.path.dirname(oufn))
    subprocess.call("msopck %s %s %s"%(msopck_oufn,data_oufn,os.path.basename(oufn)), shell=True)
    os.chdir(old)
    if not keep_temp:
        os.unlink(msopck_oufn)
        os.unlink(data_oufn)

