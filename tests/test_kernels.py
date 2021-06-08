"""Tests for kernels module"""
import pytest

from libera_sdp import kernels


def test_test_function():
    assert kernels.simple_test_function() == -9
