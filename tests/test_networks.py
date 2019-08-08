#!/usr/bin/env python3
import pytest
import machinelearning


@pytest.fixture(scope="module")
def network():
    network = machinelearning.networks
    nn = network.NeuralNetwork()


def test_network(network):
    assert True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
