"""
Decorator functions for libera_sdp
"""
# Standard
import functools
# Installed
import os
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError
# Local
from libera_sdp.config import config


def ensure_spice(f_py: callable = None, time_kernels_only: bool = False):
    """
    Before trying to understand this piece of code, read this:
    https://stackoverflow.com/questions/5929107/decorators-with-parameters/60832711#60832711

    Decorator/wrapper that tries to ensure that a metakernel is furnished in as complete a way as possible.

    ======================
    Control flow overview:
    ======================
    1. Try simply calling the wrapped function naively.
    --> SUCCESS? Great! We're done.
    --> SpiceyError? Go to step 2.

    2. Furnish metakernel at SPICE_METAKERNEL
    --> SUCCESS? Great, return the original function again (so it can be re-run).
    --> KeyError? Seems like SPICE_METAKERNEL isn't set, no problem. Go to step 3.

    ======
    Usage:
    ======
    Three ways to use this object:
    1) A decorator with no arguments
    ```
    @ensure_spice
    def my_spicey_func(a, b):
        pass
    ```
    2) A decorator with parameters. This is useful
    if we only need the latest SCLK and LSK kernels for the function involved.
    ```
    @ensure_spice(time_kernels_only=True)
    def my_spicey_time_func(a, b):
        pass
    ```
    3) An explicit wrapper function, providing dynamic parameters to the SDC API call.
    ```
    wrapped = ensure_spice(spicey_func, time_kernels_only=True)
    result = wrapped(*args, **kwargs)
    ```

    Parameters
    ----------
    f_py: callable
        The function requiring SPICE that we are going to wrap if being used explicitly,
        Otherwise None, in which case ensure_spice is being used, not as a function wrapper (see l2a_processing.py) but
        as a true decorator without an explicit function argument.
    time_kernels_only: bool, optional
        Specify that we only need to furnish time kernels
        (if SPICE_METAKERNEL is set, we still just furnish that metakernel.

    Returns
    -------
    : callable
        Decorated function, with spice error handling
    """
    assert callable(f_py) or f_py is None  # If this is set, it must be a callable object

    def _decorator(func):
        """This is either a decorator or a function wrapper, depending on how ensure_spice is being used"""

        @functools.wraps(func)
        def wrapper_ensure_spice(*args, **kwargs):
            """
            This function wraps the actual function that ensure_spice is wrapping/decorating. *args and **kwargs
            refer to those passed to the decorated function.
            """
            try:
                # Step 1.
                return func(*args, **kwargs)  # Naive first try. Maybe SPICE is already furnished.
            except SpiceyError as spcy_err:
                try:
                    # Step 2.
                    metakernel_path = os.environ['SPICE_METAKERNEL']
                    spice.furnsh(metakernel_path)
                except KeyError:
                    if time_kernels_only:
                        spice.furnsh(config.get('FALLBACK_LSK'))
                        spice.furnsh(config.get('JPSS_SCLK'))
                    else:
                        raise SpiceyError(f"When calling a function requiring SPICE, we failed to load a metakernel. "
                                          f"SPICE_METAKERNEL is not set, and time_kernels_only is not set to True"
                                          ) from spcy_err
                return func(*args, **kwargs)
        return wrapper_ensure_spice
    return _decorator(f_py) if callable(f_py) else _decorator
