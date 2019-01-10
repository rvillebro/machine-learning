#!/usr/bin/env python3
from activation_functions import ActivationFunction, ReLU
import numpy as np
from numpy import random

class Layer():
    def __init__(self):
        raise NotImplementedError('__init__ not implemented in {} class'.format(slef.__class__.__name__))
    
    def _initialize(self):
        raise NotImplementedError('_initialize not implemented in {} class'.format(slef.__class__.__name__))

    def _forward(self):
        raise NotImplementedError('_forward not implemented in {} class'.format(slef.__class__.__name__))

    def _backward(self):
        raise NotImplementedError('_backward not implemented in {} class'.format(slef.__class__.__name__))
    
    def __str__(self):
        ret_str = '{} layer with {} node(s) and {} as activation function'.format(self.__class__.__name__, self.nodes, self.activation_function.__class__.__name__)
        if self.bias:
            ret_str += ' and bias'
        return ret_str

class Dense(Layer):
    def __init__(self, nodes: int, bias: bool = True, activation_function: ActivationFunction = ReLU()):
        self.nodes = nodes
        self.bias = bias
        self.activation_function = activation_function
        self.previous_layer = None
        self.next_layer = None
        self._weights = None
        self._inputs = None
        self._wx = None
        self._outputs = None
        self._derivatives = None

    def _initialize(self):
        # if layer is the input layer weights are not initialized
        if self.previous_layer is not None:
            nweights = self.previous_layer.nodes
            # adds bias weight
            if self.bias:
                nweights += 1
            self._weights = random.ranf((self.nodes, nweights))

    def _forward(self):
        self._inputs = self.previous_layer._outputs
        # add input bias column
        if self.bias:
            bias_column = np.ones((self._inputs.shape[0], 1))
            self._inputs = np.append(self._inputs, bias_column, axis= 1)

        # calculates outputs
        self._wx = np.matmul(self._inputs, self._weights.T)
        self._outputs = self.activation_function.result(self._wx)
        return self._outputs

    def _backward(self):
        if self.next_layer is None:
            self._derivatives = self.activation_function.derivative(self._wx) * self._derivatives # delta_afunc / delta_wx  *  delta_E / delta_afunc
        else:
            #afunc_derivative = self.activation_function.derivative(self._wx)

            print(type(self.next_layer._weights), type(self.next_layer._derivatives))
            #etotal = np.matmul(self.next_layer._weights.T, self.next_layer._derivatives.T).T
            #print(self.next_layer._weights)
            #print(self.next_layer._derivatives)
            #print(etotal)
