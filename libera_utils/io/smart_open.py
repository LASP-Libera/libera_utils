"""Module for smart_open"""
# Standard
from gzip import GzipFile
from pathlib import Path
import warnings
# Installed
from cloudpathlib import S3Path, AnyPath


def is_s3(path: str or Path or S3Path):
    """Determine if a string points to an s3 location or not.

    Parameters
    ----------
    path : str or Path or S3Path
        Path to determine if it is and s3 location or not.

    Returns
    -------
    : bool
    """

    if isinstance(path, str):
        return path.startswith('s3://')
    if isinstance(path, Path):
        if str(path).startswith('s3://'):
            warnings.warn("Path object appears to contain an S3 path. "
                          "You should use S3Path to refer to S3 object urls.")
        return False
    if isinstance(path, S3Path):
        return True
    raise ValueError(f"Unrecognized path type for {path} ({type(path)})")


def is_gzip(path: str or Path or S3Path):
    """Determine if a string points to an gzip file.

    Parameters
    ----------
    path : str or Path or S3Path
        Path to check.

    Returns
    -------
    : bool
    """
    if isinstance(path, str):
        return path.endswith('.gz')
    return path.name.endswith('.gz')


def smart_open(path: str or Path or S3Path, mode: str = 'rb', enable_gzip: bool = True):
    """
    Open function that can handle local files or files in an S3 bucket. It also
    correctly handles gzip files determined by a `*.gz` extension.

    Parameters
    ----------
    path : str or Path or S3Path
        Path to the file to be opened. Files residing in an s3 bucket must begin
        with "s3://".
    mode: str, Optional
        Optional string specifying the mode in which the file is opened. Defaults
        to 'rb'.
    enable_gzip : bool, Optional
        Flag to specify that `*.gz` files should be opened as a `GzipFile` object.
        Setting this to False is useful when creating the md5sum of a `*.gz` file.
        Defaults to True.

    Returns
    -------
    : filelike object
    """
    def _gzip_wrapper(fileobj):
        """Wrapper around a filelike object that unzips it
        (if it is enabled and if the file object was opened in binary mode).

        Parameters
        ----------
        fileobj : filelike object
            The original (possibly zipped) object

        Returns
        -------
        : GzipFile or filelike object
        """
        if is_gzip(path) and enable_gzip:
            if 'b' not in mode:
                raise IOError(f'Gzip files must be opened in binary (b) mode. Got {mode}.')
            return GzipFile(filename=path, fileobj=fileobj)
        return fileobj

    if isinstance(path, (Path, S3Path)):
        return _gzip_wrapper(path.open(mode=mode))

    # AnyPath is polymorphic to Path and S3Path. Disable false pylint error
    return _gzip_wrapper(AnyPath(path).open(mode=mode))  # pylint: disable=E1101
