#!/usr/bin/env python3
import pytest


def test_version():
    from machinelearning import __version__

    assert __version__ == "0.1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
