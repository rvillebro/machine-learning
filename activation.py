#!/bin/python
import math
import numpy as np

class ActivationFunction():
    __slots__ = ['function', 'function_prime']
    
    def fire(self, inputs):
        try:
            return self.function(inputs)
        except AttributeError:
            raise AttributeError('Activation function not defined')

    def gradient(self, inputs):
        try:
            return self.function_prime(inputs)
        except AttributeError:
            raise AttributeError('Activation function not defined')


class Sigmoid(ActivationFunction):
    def __init__(self):
        self.function = lambda inputs: 1 / (1 + math.e ** - inputs)

class TanH(ActivationFunction):
    def __init__(self):
        self.function = lambda inputs: (math.e ** inputs - math.e ** - inputs) / (math.e ** inputs + math.e ** - inputs)

class ReLU(ActivationFunction):
    def ReLU(self, inputs):
        inputs[inputs < 0] = 0
        return inputs
    
    def ReLU_prime(self, inputs):
        inputs[inputs < 0] = 0
        inputs[inputs > 0] = 1
        return inputs

    def __init__(self):
        self.function = self.ReLU
        self.function_prime = self.ReLU_prime

class SoftMax(ActivationFunction):
    def softmax(self, inputs):
        esum = (math.e ** inputs).sum()
        return [math.e ** input / esum for input in inputs]
        
    def __init__(self):
        self.function = self.softmax