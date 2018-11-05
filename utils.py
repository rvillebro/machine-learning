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
        shuffle_index = permutation(self.X.shape[0])
        self.X = self.X[shuffle_index]
        self.y = self.y[shuffle_index]