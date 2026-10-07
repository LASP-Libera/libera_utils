"""Tests for the autouse ``block_outbound_network`` fixture in ``tests/conftest.py``.

Every test in this module runs under the guard, so each one exercises it directly. The remote
addresses are from the documentation ranges (RFC 5737 and RFC 3849) and the reserved ``.invalid``
domain (RFC 6761), so if the guard regresses these tests fail without reaching a real host, and
the timeout bounds how long an unrouted connection attempt can hang.
"""

import re
import socket

import pytest


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
@pytest.mark.parametrize(
    ("family", "address"),
    [
        (socket.AF_INET, ("192.0.2.1", 9)),
        (socket.AF_INET6, ("2001:db8::1", 9, 0, 0)),
        (socket.AF_INET, ("libera.invalid", 443)),
    ],
    ids=["ipv4", "ipv6", "hostname"],
)
def test_a_remote_connection_fails_the_test(request, family, address, method):
    """A connection to a remote host raises, naming the offending test, for both connect methods."""
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        with pytest.raises(RuntimeError, match=re.escape(request.node.nodeid)):
            getattr(sock, method)(address)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_a_loopback_connection_is_allowed(host):
    """Loopback stays reachable, so a local server such as moto in server mode still works."""
    with socket.create_server(("127.0.0.1", 0)) as server, socket.socket() as client:
        client.connect((host, server.getsockname()[1]))
        connection, _ = server.accept()
        connection.close()
