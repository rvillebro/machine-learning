#!/bin/python
import numpy as np


class LossFunction():
    __slots__ = ['function', 'function_prime']
    
    def loss(self, y, y_hat):
        try:
            return self.function(y, y_hat)
        except AttributeError:
            raise AttributeError('Loss function not defined')
    
    def gradient(self, y, y_hat):
        try:
            return self.function_prime(y, y_hat)
        except AttributeError:
            raise AttributeError('Loss prime function not defined')



class MSE(LossFunction):
    def __init__(self):
        self.function = lambda y, y_hat: 1 / y.size * ((y - y_hat) ** 2).sum()
        self.function_prime = lambda y, y_hat: y_hat - y
