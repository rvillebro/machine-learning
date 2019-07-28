#!/usr/bin/env python3
import pytest


def test_networks():
    from machinelearning import __version__

    assert __version__ == "0.1"
    return


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
