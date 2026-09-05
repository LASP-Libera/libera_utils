"""Shared pytest marks."""

import pytest

# "error" makes any warning a test provokes a failure, which is the point where a test asserts
# that a well-formed input produces no warnings. The second filter carves out one exception.
#
# PytestUnraisableExceptionWarning does not report a warning raised by the test it is attributed
# to. pytest drains unraisable exceptions -- typically a ResourceWarning from a file that the
# garbage collector finalized while it was still open -- from a single global queue at every
# test's setup, call and teardown. The blame therefore lands on whichever test happens to be
# running when the collector fires, which is usually not the test that leaked the file. Under a
# bare "error" filter that turned any leaked file handle in the suite into a failure of an
# arbitrary marked test, on an arbitrary Python version.
#
# A leak belongs to the code that opened the file, so let these surface as warnings rather than
# fail an unrelated test. Run the suite with `-W error::ResourceWarning` to hunt for leaks.
strict_warnings = pytest.mark.filterwarnings("error", "default::pytest.PytestUnraisableExceptionWarning")
