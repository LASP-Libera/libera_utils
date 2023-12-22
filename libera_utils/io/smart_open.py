"""Module for smart_open"""
# Standard
from gzip import GzipFile
from pathlib import Path
import warnings
import shutil
# Installed
import boto3
from cloudpathlib import S3Path, AnyPath


def is_s3(path: str or Path or S3Path):
    """Determine if a string points to an s3 location or not.

    Parameters
    ----------
    path : str or pathlib.Path or cloudpathlib.s3.s3path.S3Path
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
    path : str or pathlib.Path or cloudpathlib.s3.s3path.S3Path
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
    path : str or pathlib.Path or cloudpathlib.s3.s3path.S3Path
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
    : typing.IO or gzip.GzipFile
    """
    def _gzip_wrapper(fileobj):
        """Wrapper around a filelike object that unzips it
        (if it is enabled and if the file object was opened in binary mode).

        Parameters
        ----------
        fileobj : typing.IO
            The original (possibly zipped) object

        Returns
        -------
        : gzip.GzipFile
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


def _copy_local_to_local(source_path: str or Path, dest_path: str or Path, delete: bool):
    """Copy a local source file to a local destination.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Path to the source file to be copied.
    dest_path : str or pathlib.Path
        Path to the destination for the copied file.
    delete : bool
        If true, deletes files copied from source (default = False)

    Returns
    -------
    : pathlib.Path
        The path to the newly created file
    """
    # This is a local copy and uses shutil copy
    local_source_path = Path(source_path)
    local_dest_path = Path(dest_path)

    # Warning if no suffix is used in destination.
    if len(local_dest_path.suffix) == 0:
        warnings.warn(f'You have copied to a location without a file extension.'
                      f'Source location: {local_source_path} to destination:'
                      f'{local_dest_path}.')

    # Returns a PosixPath of the newly created file
    if delete:
        return shutil.move(source_path, dest_path)

    return shutil.copy(source_path, dest_path)


def _copy_local_to_s3(source_path: str or Path, dest_path: str or S3Path, delete: bool):
    """Copy a local file to an S3 object.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Path to the source file to be copied.
    dest_path : str or cloudpathlib.s3.s3path.S3Path
        Path to the destination for the copied file. Files residing in an s3 bucket
        must begin with "s3://".
    delete : bool
        If true, deletes files copied from source (default = False)

    Returns
    -------
    : cloudpathlib.s3.s3path.S3Path
        The path to the newly created file
    """
    # This is a local to remote copy and uses S3 upload
    s3_dest_path = S3Path(dest_path)
    local_source_path = Path(source_path)

    # Warning if no suffix is used.
    if len(s3_dest_path.suffix) == 0:
        warnings.warn(f'You have copied a file to S3 without a file extension.'
                      f'Source location: {local_source_path} to S3 location:'
                      f'{s3_dest_path}.')

    s3 = boto3.resource("s3")
    # Has no return, but will raise exceptions on problems
    s3.Bucket(s3_dest_path.bucket).upload_file(str(local_source_path), s3_dest_path.key)
    if delete:
        local_source_path.unlink()
    return s3_dest_path


def _copy_s3_to_local(source_path: str or Path, dest_path: str or S3Path, delete: bool):
    """Copy an S3 object to a local file.

    Parameters
    ----------
    source_path : str or cloudpathlib.s3.s3path.S3Path
        Path to the source file to be copied. Files residing in an s3 bucket must begin
        with "s3://".
    dest_path : str or pathlib.Path
        Path to the destination for the copied file.
    delete : bool
        If true, deletes files copied from source (default = False)

    Returns
    -------
    : pathlib.Path
        The path to the newly created file
    """
    # This is a remote to local copy and uses S3 download
    s3_source_path = S3Path(source_path)
    local_dest_path = Path(dest_path)

    # Ensure a full destination path including file name is used
    if local_dest_path.is_dir():
        local_dest_path = local_dest_path / s3_source_path.name
        warnings.warn(f'A directory was given as the destination for the smart file '
                      f'copy. This was modified to include a name as follows.'
                      f'Copy from {s3_source_path} to {local_dest_path}.')

    # Warning if no suffix is used.
    if len(local_dest_path.suffix) == 0:
        warnings.warn(f'You have copied a file without a file extension.'
                      f'Source: {s3_source_path} to destination:'
                      f'{local_dest_path}.')

    s3 = boto3.resource("s3")
    # Has no return, but will raise exceptions on problems
    s3.Bucket(s3_source_path.bucket).download_file(s3_source_path.key, str(local_dest_path))
    if delete:
        s3.Object(s3_source_path.bucket, s3_source_path.key).delete()
    return Path(local_dest_path)


def _copy_s3_to_s3(source_path: str or Path, dest_path: str or S3Path, delete: bool):
    """Copy an S3 object to a different S3 object.

    Parameters
    ----------
    source_path : str or cloudpathlib.s3.s3path.S3Path
        Path to the source file to be copied. Files residing in an s3 bucket must begin
        with "s3://".
    dest_path : str or cloudpathlib.s3.s3path.S3Path
        Path to the Destination file to be copied to. Files residing in an s3 bucket
        must begin with "s3://".
    delete : bool
        If true, deletes files copied from source (default = False)

    Returns
    -------
    : cloudpathlib.s3.s3path.S3Path
        The path to the newly created file
    """
    # This is a remote to remote copy and uses S3 copy
    s3_source_path = S3Path(source_path)
    s3_dest_path = S3Path(dest_path)

    copy_source = {
        'Bucket': s3_source_path.bucket,
        'Key': s3_source_path.key
    }

    # Warning if no suffix is used.
    if len(s3_dest_path.suffix) == 0:
        warnings.warn(f'You have copied a file to S3 without a file extension.'
                      f'Source location: {s3_source_path} to S3 location:'
                      f'{s3_dest_path}.')

    client = boto3.client("s3")
    # Has no return, but will raise exceptions
    client.copy(copy_source, s3_dest_path.bucket, s3_dest_path.key)

    if delete:
        s3 = boto3.resource("s3")
        s3.Object(copy_source['Bucket'], copy_source['Key']).delete()
    return s3_dest_path


def smart_copy_file(source_path: str or Path or S3Path, dest_path: str or Path or S3Path, delete=False):
    """Copy function that can handle local files or files in an S3 bucket.
    Returns the path to the newly created file as a Path or an S3Path, depending on the destination.

    Parameters
    ----------
    source_path : str or pathlib.Path or cloudpathlib.s3.s3path.S3Path
        Path to the source file to be copied. Files residing in an s3 bucket must begin
        with "s3://".
    dest_path : str or pathlib.Path or cloudpathlib.s3.s3path.S3Path
        Path to the Destination file to be copied to. Files residing in an s3 bucket
        must begin with "s3://".
    delete : bool
        If true, deletes files copied from source (default = False)

    Returns
    -------
    : pathlib.Path or cloudpathlib.s3.s3path.S3Path
        The path to the newly created file
    """
    if not is_s3(source_path) and not is_s3(dest_path):
        return _copy_local_to_local(source_path, dest_path, delete)

    if is_s3(dest_path) and not is_s3(source_path):
        return _copy_local_to_s3(source_path, dest_path, delete)

    if is_s3(source_path) and not is_s3(dest_path):
        return _copy_s3_to_local(source_path, dest_path, delete)

    return _copy_s3_to_s3(source_path, dest_path, delete)
