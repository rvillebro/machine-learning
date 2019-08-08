#!/bin/python
import numpy as np
import random


def shuffle(x, y):
    tmp = list(zip(x, y))
    random.shuffle(tmp)
    x, y = zip(*tmp)


def batch(x, y, batch_size=10):
    for i in range(0, x.shape[0], batch_size):
        yield (x[i:i + batch_size], y[i:i + batch_size])
