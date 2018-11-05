#!/bin/python
import numpy as np
from numpy.random import permutation

class DataSet():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def normalize(self):
        pass

    def shuffle(self):
        pass