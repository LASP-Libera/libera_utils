"""
SPICE Kernel Manager

This module provides a clean interface for managing SPICE kernels for the Libera instrument.
It separates kernel lifecycle management from computation logic to improve maintainability and performance.
"""

import datetime
import logging
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path

from cloudpathlib import AnyPath
from curryer import meta
from curryer import spicierpy as sp

from libera_utils import kernel_maker
from libera_utils.config import config
from libera_utils.io.caching import get_local_short_temp_dir, validate_path_length
from libera_utils.libera_spice.spice_utils import (
    NAIF_DE_REGEX,
    NAIF_HIGH_PREC_PCK_REGEX,
    NAIF_LSK_REGEX,
    NAIF_PCK_REGEX,
    KernelFileCache,
    find_most_recent_naif_kernel,
    ls_kernels,
)

logger = logging.getLogger(__name__)

# Ensure the leap second file in libera_utils is used by curryer for all kernel making operations
# TODO[CURRYER-97]: This environment variable must be set or a leapsecond kernel must be in a specific relative
# path location for curryer to make any kernels. This obscures the process and should be re-evaluated and improved
# in the future.
# os.environ["LEAPSECOND_FILE_ENV"] = config.get("GENERIC_KERNEL_DIR")


# TODO [LIBSDC-687] This class should likely be in curryer instead of libera_utils
class KernelManager:
    """
    Manages SPICE kernel loading and lifecycle for Libera geolocation calculations.

    Parameters
    ----------
    temp_dir_base : str | Path | None
        Base directory for temporary kernel files. If None, uses platform default.
        Can also be controlled via LIBERA_TEMP_DIR environment variable.
    max_path_length : int
        Maximum allowed path length when validation is enabled (default: 80 to match SPICE requirements).
    """

    _naif_kernel_regexs = [
        NAIF_DE_REGEX,
        NAIF_PCK_REGEX,
        NAIF_LSK_REGEX,
    ]

    _high_precision_earth_regexs = [
        r"earth_assoc_itrf93\.tf",
        NAIF_HIGH_PREC_PCK_REGEX,
    ]

    _max_path_length: int = 80

    def __init__(
        self,
        temp_dir_base: str | Path | None = None,
        download_naif_url: str = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/",
        use_test_naif_url: bool = False,
        use_high_precision_earth: bool = True,
        cache_timeout_days: int = 7,
    ):
        """Initialize the kernel manager with no kernels furnished."""
        self._loaded_kernels: sp.ext.load_kernel | None = None
        self._static_loaded: bool = False
        self._dynamic_loaded: bool = False
        self._naif_kernels_loaded: bool = False
        self._high_precision_earth_loaded: bool = False
        self._static_kernels_path: str | AnyPath | None = None
        self._naif_kernels_path: str | AnyPath | None = None

        self._use_high_precision_earth: bool = use_high_precision_earth
        self._naif_download_url = download_naif_url
        self._use_test_naif_url = use_test_naif_url

        # Configure temporary directory base
        self._temp_base: Path = Path(temp_dir_base) if temp_dir_base else get_local_short_temp_dir()

        self._cache_timeout_days: int = datetime.timedelta(days=cache_timeout_days)

        logger.debug(f"Using temp base directory: {self._temp_base}")

    # TODO[LIBSDC-704]: Adding caching of these static kernels to avoid re-creation on each run
    def _create_temporary_static_kernels(self) -> Path:
        """
        Curryer uses json based kernel configuration files to generate SPICE kernels. These configuration files need
        to be processed into actual SPICE kernel files before they can be loaded. This method creates a temporary
        directory, copies the necessary configuration files, and generates the static kernels into that directory.

        This method will start by furnishing the naif kernels to ensure the leap second kernel is available,
        as it is required for static kernel generation.

        Returns
        -------
        Path
            Path to the created temporary directory containing static kernels.

        Raises
        ------
        RuntimeError
            If kernel creation fails or path exceeds length limit.
        """
        try:
            # Create a unique temporary directory with short prefix
            # Use a short, recognizable prefix
            temp_dir = tempfile.mkdtemp(dir=str(self._temp_base))
            temp_path = Path(temp_dir)

            # Validate path length if enabled
            validate_path_length(temp_path, KernelManager._max_path_length)

            # Furnish the leap second kernel first
            # TODO[CURRYER-97]: This is required for curryer kernel making to work, but should be improved in the future
            # potentially with caching or tracking explicitly of the leap second kernel by the KernelManager
            if not self._naif_kernels_loaded:
                self.load_naif_kernels()

            logger.info(f"Creating static kernels in: {temp_path} (length: {len(str(temp_path))})")

            # Copy mission level static libera kernels to temp directory
            libera_kernels_path = Path(config.get("LIBERA_KERNEL_DIR"))
            for file in libera_kernels_path.iterdir():
                logger.debug(f"Copying mission kernel {file} to temp directory {temp_path}")
                shutil.copy(file, temp_path / file.name)

            # Ensure instrument kernel path is valid in length
            instrument_kernel_path = Path(config.get("LIBERA_KERNEL_INSTRUMENT"))
            validate_path_length(temp_path / instrument_kernel_path.name, KernelManager._max_path_length)

            # Generate kernels from configuration files
            for kernel_config_file in config.get("LIBERA_KERNEL_STATIC_CONFIGS"):
                config_path = Path(kernel_config_file)
                if not config_path.is_file():
                    raise FileNotFoundError(f"Kernel config not found: {kernel_config_file}")

                # Copy the instrument static file into the temp directory for processing
                logger.debug(f"Copying kernel config {config_path} to temp directory {temp_path}")
                shutil.copy(config_path, temp_path / config_path.name)

                # Generate kernel
                kernel_maker.make_kernel(temp_path / config_path.name, temp_path, input_data=None)

            # Verify kernels were created
            created_files = list(temp_path.iterdir())
            if not created_files:
                raise RuntimeError(f"No kernel files created in {temp_path}")

            logger.info(f"Created {len(created_files)} static kernel files")
            return temp_path

        except Exception as e:
            # Clean up temp directory if kernel creation failed
            if "temp_path" in locals() and temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
            logger.error(f"Failed to create static kernels: {e}")
            raise RuntimeError(f"Static kernel creation failed: {e}") from e

    def _delete_temporary_static_kernels(self) -> None:
        """
        Delete the temporary static kernels directory.

        Safely removes the directory and all contents. Logs warnings
        but doesn't raise exceptions to allow cleanup to proceed.
        """
        if self._static_kernels_path is None:
            logger.debug("No static kernels path to delete")
            return

        try:
            if self._static_kernels_path.exists():
                shutil.rmtree(self._static_kernels_path)
                logger.info(f"Deleted static kernels directory: {self._static_kernels_path}")
            else:
                logger.debug(f"Static kernels path already removed: {self._static_kernels_path}")
        except Exception as e:
            # Log but don't raise - we want cleanup to continue
            warnings.warn(f"Failed to delete static kernels directory: {e}")
        finally:
            self._static_kernels_path = None

    def load_static_kernels(self) -> None:
        """
        Load Libera static kernels into SPICE.

        Static kernels include instrument orientation and configuration data
        that doesn't change with time. This only needs to be called once.

        Raises
        ------
        FileNotFoundError
            If kernel files cannot be found at expected paths.
        RuntimeError
            If kernel loading fails or paths exceed length limits.
        """
        if self._static_loaded:
            logger.debug("Static kernels already loaded, skipping")
            return

        try:
            # Create the static kernels if not already created
            if self._static_kernels_path is None:
                self._static_kernels_path = self._create_temporary_static_kernels()

            # Load metakernel
            metakernel = meta.MetaKernel.from_json(
                config.get("LIBERA_KERNEL_META"),
                relative=False,
                sds_dir=config.get("GENERIC_KERNEL_DIR"),
                mission_dir=config.get("LIBERA_KERNEL_DIR"),
            )

            # Get static kernel files
            static_kernels = []
            for f in self._static_kernels_path.iterdir():
                if f.is_file() and "json" not in f.suffix.lower():
                    static_kernels.append(str(f))

            if not static_kernels:
                raise FileNotFoundError(f"No static kernels found in {self._static_kernels_path}")

            # Load all kernels
            if self._loaded_kernels is None:
                self._loaded_kernels = sp.ext.load_kernel(
                    [metakernel.sds_kernels, metakernel.mission_kernels, static_kernels]
                )
            else:
                self._loaded_kernels._iter_load([metakernel.sds_kernels, metakernel.mission_kernels, static_kernels])

            self._static_loaded = True
            logger.info(f"Successfully loaded {len(static_kernels)} static kernels")

        except Exception as e:
            logger.error(f"Failed to load static kernels: {e}")
            # Clean up on failure
            self._delete_temporary_static_kernels()
            raise

    def load_naif_kernels(self, cache_time_out: int | None = None) -> None:
        """
        Load NAIF generic kernels into SPICE. This method will first look in the generic kernel directory, and if no
        local versions of the needed kernels are found, it will download them from the NAIF server and cache them
        locally.

        These include leap seconds, planetary ephemeris, and Earth orientation data.

        Raises
        ------
        FileNotFoundError
            If kernel files cannot be found at expected paths.
        RuntimeError
            If kernel loading fails.
        """
        if self._naif_kernels_loaded:
            logger.debug("NAIF kernels already loaded, skipping")
            return

        naif_cache_timeout = self._cache_timeout_days
        if cache_time_out is not None:
            naif_cache_timeout = datetime.timedelta(days=cache_time_out)

        needed_naif_kernels = KernelManager._naif_kernel_regexs.copy()
        if self._use_high_precision_earth:
            needed_naif_kernels.extend(KernelManager._high_precision_earth_regexs)

        naif_kernel_paths = []

        # First check the generic kernel directory for local versions of needed NAIF kernels
        local_kernel_dir = Path(config.get("GENERIC_KERNEL_DIR"))
        for file in local_kernel_dir.iterdir():
            if file.is_file() and any(re.match(pattern, file.name) for pattern in needed_naif_kernels):
                logger.debug(f"Found local NAIF kernel: {file.name}")
                naif_kernel_paths.append(str(file))
                # Remove from needed list
                needed_naif_kernels = [pattern for pattern in needed_naif_kernels if not re.match(pattern, file.name)]

        # Download any missing NAIF kernels from the NAIF server using the caching tools in spice_utils
        for pattern in needed_naif_kernels:
            naif_url = self._naif_download_url
            if "tpc" in pattern or "bpc" in pattern:
                naif_url = naif_url + "pck/"
            elif "tf" in pattern:
                naif_url = naif_url + "fk/planets/"
            elif "bsp" in pattern:
                naif_url = naif_url + "spk/planets/"
            elif "tls" in pattern:
                naif_url = naif_url + "lsk/"
            else:
                logger.error(f"Unknown NAIF kernel type for pattern: {pattern}. Please check the pattern.")
                raise ValueError("Unknown NAIF kernel type in pattern {pattern}, cannot determine download URL.")

            # For testing to not spam the NAIF servers we use a test URL that must be passed in and will not have the same
            # directory structure as the main NAIF server
            if self._use_test_naif_url:
                logger.warning(
                    "The use_test_naif_url flag is set to True. No directory structure will be assumed "
                    "for NAIF kernel downloads."
                )
                naif_url = self._naif_download_url

            found_file = find_most_recent_naif_kernel(naif_url, pattern)
            file_cache = KernelFileCache(found_file, max_cache_age=naif_cache_timeout)
            naif_kernel_paths.append(str(file_cache.kernel_path))

        # Load NAIF kernels
        if self._loaded_kernels is None:
            self._loaded_kernels = sp.ext.load_kernel(naif_kernel_paths)
        else:
            for kernel_path in naif_kernel_paths:
                self._loaded_kernels.load(kernel_path)

        # Set leap second file environment variable for curryer usage
        # TODO[CURRYER-97]: This is required for curryer kernel making to work when libera_utils is imported,
        #  but should be improved in the future
        lsk_path = [Path(p).parent for p in naif_kernel_paths if re.match(NAIF_LSK_REGEX, Path(p).name)]
        if len(lsk_path) == 0:
            raise RuntimeError("No leap second kernel loaded, cannot set LEAPSECOND_FILE_ENV")
        os.environ["LEAPSECOND_FILE_ENV"] = str(lsk_path[0])

        self._naif_kernels_loaded = True
        logger.info(f"Successfully loaded {len(naif_kernel_paths)} NAIF kernels")

    def load_libera_dynamic_kernels(
        self,
        dynamic_kernel_directory: str | AnyPath,
        needs_static_kernels: bool = True,
        needs_naif_kernels: bool = True,
    ) -> None:
        """
        Load dynamic kernels from a directory. This will load both static and NAIF kernels if they have not
        already been loaded.

        Dynamic kernels contain time-varying data like spacecraft or instrument position
        and attitude for a given observation period.

        Parameters
        ----------
        dynamic_kernel_directory : str | AnyPath | None
            Path to directory containing all kernel files. All files must be present in this directory,
            subdirectories will not be searched.
        needs_static_kernels : bool
            Whether to ensure static kernels are loaded before loading dynamic kernels (default: True).
        needs_naif_kernels : bool
            Whether to ensure NAIF kernels are loaded before loading dynamic kernels (default: True).
        Raises
        ------
        FileNotFoundError
            If kernel directories are empty or invalid.
        """
        if not self._static_loaded and needs_static_kernels:
            logger.info("Static kernels not loaded, loading now...")
            self.load_static_kernels()

        if not self._naif_kernels_loaded and needs_naif_kernels:
            logger.info("NAIF kernels not loaded, loading now...")
            self.load_naif_kernels()

        try:
            # This loads all files from a single directory, no subdirectories.
            dynamic_kernel_paths = [f for f in Path(dynamic_kernel_directory).iterdir() if f.is_file()]

            if not dynamic_kernel_paths:
                raise FileNotFoundError("No kernel files found in provided paths")

            self._loaded_kernels._iter_load(dynamic_kernel_paths)
            self._dynamic_loaded = True
            logger.info(f"Successfully loaded {len(dynamic_kernel_paths)} dynamic kernel files")

        except Exception as e:
            logger.error(f"Failed to load dynamic kernels: {e}")
            raise

    def unload_all(self) -> None:
        """
        Unload all SPICE kernels and clean up resources.

        This should be called when done with calculations to free memory
        and ensure clean state for subsequent operations.
        """
        if self._loaded_kernels is not None:
            try:
                # This will unload all loaded kernels from SPICE
                sp.kclear()
                logger.info("Successfully unloaded all kernels")
            except Exception as e:
                warnings.warn(f"Error during kernel unload: {e}")
            finally:
                self._loaded_kernels = None
                self._static_loaded = False
                self._dynamic_loaded = False
                self._naif_kernels_loaded = False

    def ensure_known_kernels_are_furnished(self) -> None:
        """
        Method to verify all known kernels are furnished by SPICE.

        Raises
        ------
        RuntimeError
            If kernels are not properly loaded.
        """
        furnished_kernels = {k.file_name for k in ls_kernels()}

        if len(furnished_kernels) == 0:
            warnings.warn("No kernels are currently furnished by SPICE.")
            return

        loaded_kernel_paths = {k for k in self._loaded_kernels.loaded}

        if loaded_kernel_paths == furnished_kernels:
            # All expected kernels are furnished correctly
            return

        # Case 1 - More kernels expected than furnished
        if missing := loaded_kernel_paths - furnished_kernels:
            raise RuntimeError(
                f"Not all kernels are furnished by SPICE. The following kernels were expected"
                f"but are not furnished: {missing}"
            )
        # Case 2 - More kernels furnished than expected
        if extras := furnished_kernels - loaded_kernel_paths:
            warnings.warn(
                f"More kernels are furnished by SPICE than expected. The following extra kernels are present: {extras}"
            )
            # All expected kernels are furnished, so we consider this a success
            return

    def __enter__(self):
        """Enter context manager, loading static kernels automatically."""
        self.load_static_kernels()
        self.load_naif_kernels()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, unloading all kernels."""
        try:
            self.unload_all()
        finally:
            self._delete_temporary_static_kernels()
        return False

    def __del__(self):
        """Cleanup temporary files when object is garbage collected."""
        try:
            self.unload_all()
            self._delete_temporary_static_kernels()
        except Exception:
            logger.warning("No static kernels to delete during cleanup")
            pass
