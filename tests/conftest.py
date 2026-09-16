"""Pytest fixtures"""

import logging
import socket

import pytest

pytest_plugins = [
    "tests.plugins.data_path_fixtures",
    "tests.plugins.data_product_fixtures",
    "tests.plugins.spice_fixtures",
    "tests.plugins.aws_fixtures",
    "tests.plugins.manifest_fixtures",
    "tests.plugins.integration_test_fixtures",
    "tests.plugins.l1a_fixtures",
]


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Provides a monkeypatch that applies for an entire pytest session (saves time)"""
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture
def cleanup_loggers():
    """Ensures that root logging handlers are removed after a test"""
    yield
    root = logging.getLogger()
    root.handlers = []


#: Loopback addresses a test may legitimately connect to (moto in server mode, local fixtures).
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.fixture(autouse=True)
def block_outbound_network(request):
    """Fail any test outside the ``e2e`` lane that opens a connection to a remote host.

    The unit and integration lanes are meant to run code, not pull data, but nothing enforced that:
    a test could mock the call it was written about and still reach the network through its setup
    path. ``KernelManager.load_static_kernels`` calls ``load_naif_kernels`` first, so patching only
    the method under test left a real NAIF download in a unit test for months, where it cost up to
    34 seconds and varied with NAIF's availability (LIBSDC-703).

    Blocking at the socket rather than at each library keeps the guard indifferent to how the call
    is made from Python -- ``requests``, ``botocore``, ``urllib`` all end up here. Mocking layers
    such as ``responses`` and ``moto`` intercept above the socket and are unaffected.

    It is process-local, and deliberately not more than that: patching ``socket.socket`` cannot see
    what a child process does. Kernel generation shells out to ``mkspk`` and ``msopck``, so a
    subprocess that reached the network would pass unnoticed. Those two read local files and are
    not a plausible route to one, but a test that shells out to something which *might* be belongs
    in ``tests/e2e`` on that basis alone.

    Tests that genuinely need a live service belong in ``tests/e2e`` and carry
    ``@pytest.mark.e2e``, which lifts the guard.
    """
    if request.node.get_closest_marker("e2e"):
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _host_of(address):
        return address[0] if isinstance(address, tuple) else address

    def _is_local(sock, address):
        return sock.family not in (socket.AF_INET, socket.AF_INET6) or _host_of(address) in _ALLOWED_HOSTS

    def _guard(real):
        def wrapper(self, address, *args, **kwargs):
            if _is_local(self, address):
                return real(self, address, *args, **kwargs)
            raise RuntimeError(
                f"{request.node.nodeid} attempted a network connection to {_host_of(address)!r}. "
                "Unit and integration tests must not contact external services -- mock the transport, "
                "or move the test to tests/e2e and mark it @pytest.mark.e2e."
            )

        return wrapper

    socket.socket.connect = _guard(real_connect)
    socket.socket.connect_ex = _guard(real_connect_ex)
    try:
        yield
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
