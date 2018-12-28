#!/usr/bin/env python3
import math
import numpy as np

class ActivationFunction():
    def result(self, inputs):
        try:
            return self.function(inputs)
        except AttributeError:
            raise AttributeError('Activation function not defined')

    def derivative(self, inputs):
        try:
            return self.function_derivative(inputs)
        except AttributeError:
            raise AttributeError('Activation function derivative not defined')


class ReLU(ActivationFunction):
    def relu_function(self, inputs):
        inputs[inputs < 0] = 0
        return inputs
    
    def relu_function_derivative(self, inputs):
        inputs[inputs < 0] = 0
        inputs[inputs > 0] = 1
        return inputs

    def __init__(self):
        self.function = self.relu_function
        self.derivative = self.relu_function_derivative



######## TO BE IMPLEMENTED ########

class Sigmoid(ActivationFunction):
    def __init__(self):
        self.function = lambda inputs: 1 / (1 + math.e ** - inputs)

class TanH(ActivationFunction):
    def __init__(self):
        self.function = lambda inputs: (math.e ** inputs - math.e ** - inputs) / (math.e ** inputs + math.e ** - inputs)

class SoftMax(ActivationFunction):
    def softmax(self, inputs):
        esum = (math.e ** inputs).sum()
        return [math.e ** input / esum for input in inputs]
        
    def __init__(self):
        self.function = self.softmax