#!/usr/bin/env python3
"""
Layers (layers)
======================

This module contain all layers available for neural networks: :class:`Dense`. All layers must be a subclass of :class:`Layer`.

.. codeauthor:: Rasmus Villebro <rasmus-villebro@hotmail.com>
"""
__version__ = 1.0
__author__ = 'Rasmus Villebro'
__email__ = 'rasmus-villebro@hotmail.com'


from activation_functions import ActivationFunction, ReLU
import numpy as np
from numpy import random

class Layer():
    """
    Superclass for all layers. Used for defining which methods must be implemented when implementing a new layer.
    """
    def __init__(self):
        raise NotImplementedError('__init__ not implemented in {} class'.format(self.__class__.__name__))

    def _initialize(self):
        raise NotImplementedError('_initialize not implemented in {} class'.format(self.__class__.__name__))

    def _forward(self):
        raise NotImplementedError('_forward not implemented in {} class'.format(self.__class__.__name__))

    def _backward(self):
        raise NotImplementedError('_backward not implemented in {} class'.format(self.__class__.__name__))
    
    def __str__(self):
        ret_str = '{} layer with {} node(s) and {} as activation function'.format(self.__class__.__name__, self.nodes, self.activation_function)
        if self.bias:
            ret_str += ' and bias'
        return ret_str

class Dense(Layer):
    """
    A fully connected layer (superclass: :class:`Layer`)
    """
    def __init__(self, nodes: int, bias: bool = True, activation_function: ActivationFunction = ReLU()):
        """
        :param int nodes: The number of nodes in the layer.
        :param bool bias: Whether to include a bias node or not.
        :param ActivationFunction activation_function: The activation function to use in the layer, must be a subclass of :class:`activation_fucntions.ActivationFunction`.
        """
        if not isinstance(nodes, int):
            raise TypeError('type {} not valid for argument nodes. Please parse an int..'.format(type(nodes)))
        if not isinstance(bias, bool):
            raise TypeError('type {} not valid for argument bias. Please parse a bool.'.format(type(bias)))
        if not isinstance(activation_function, ActivationFunction):
            raise TypeError('type {} not valid for argument activation_function. Please parse a subclass of ActivationFunction.'.format(type(activation_function)))
        
        #: Number of nodes in the layer
        self.nodes = nodes
        #: Indicates whether a bias node should be included
        self.bias = bias
        #: The activation function of the layer
        self.activation_function = activation_function
        #: The previous connected layer
        self._previous_layer = None
        #: The next connected layer
        self._next_layer = None
        #: Weights of the layer
        self._weights = None
        #: Inputs to the layer
        self._inputs = None
        #: Outputs of the layer
        self._outputs = None
        self._wx = None
        self._derivatives = None

    def _initialize(self):
        """
        Initializes weights
        """
        # if layer is the input layer weights are not initialized
        if self._previous_layer is not None:
            nweights = self._previous_layer.nodes
            # adds bias weight
            if self.bias:
                nweights += 1
            self._weights = random.ranf((self.nodes, nweights))

    def _forward(self):
        """
        Forward pass
        """
        self._inputs = self._previous_layer._outputs
        # add input bias column
        if self.bias:
            bias_column = np.ones((self._inputs.shape[0], 1))
            self._inputs = np.append(self._inputs, bias_column, axis= 1)

        # calculates outputs
        self._wx = np.matmul(self._inputs, self._weights.T)
        self._outputs = self.activation_function.result(self._wx)
        return self._outputs

    def _backward(self):
        """
        Backward pass
        """
        if self._next_layer is None:
            activation_function_derivatives = self.activation_function.derivative(self._wx) # d_afunc / d_wx
            output_derivatives = self._inputs # d_wx / d_weights
            self._weight_updates = self._derivatives * activation_function_derivatives * output_derivatives
            self._backprop_pass_on = self._derivatives * activation_function_derivatives

            self._derivatives = self.activation_function.derivative(self._wx) * self._derivatives # delta_afunc / delta_wx  *  delta_E / delta_afunc
        else:
            activation_function_derivatives = self.activation_function.derivative(self._wx) # d_afunc / d_wx
            output_derivatives = self._inputs # d_wx / d_weights
            error_derivatives = self._next_layer._weights # d_error / d_output
            self._weight_updates = self._backprop_pass_on * error_derivatives * output_derivatives * activation_function_derivatives
            print(type(self._next_layer._weights), type(self._next_layer._derivatives))
            #etotal = np.matmul(self.next_layer._weights.T, self.next_layer._derivatives.T).T
            #print(self.next_layer._weights)
            #print(self.next_layer._derivatives)
            #print(etotal)
