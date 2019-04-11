#!/usr/bin/env python3
"""
Loss functions (loss_functions)
======================================

This module contains all available loss functions for networks: :class:`MSE`, bla and bla. All loss functions must be a subclass of :class:`LossFunction`.
"""

import numpy as np

class LossFunction():
    """
    Superclass for all loss functions. Used for defining which methods must be implemented when implementing a new loss function.
    """
    def result(self, y, y_hat):
        try:
            return self.function(y, y_hat)
        except AttributeError:
            raise AttributeError('Loss function not defined')
    
    def derivative(self, y, y_hat):
        try:
            return self.function_derivative(y, y_hat)
        except AttributeError:
            raise AttributeError('Loss derivative function not defined')

    def __str__(self):
        return str(self.__class__.__name__)



class MSE(LossFunction):
    """
    MSE loss function
    """
    def __init__(self):
        self.function = lambda y, y_hat: 1 / y.size * ((y - y_hat) ** 2).sum()
        self.function_derivative = lambda y, y_hat: y_hat - y
