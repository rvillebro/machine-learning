#!/usr/bin/env python3
from ..activation_functions import ActivationFunction, ReLU
import numpy as np
from numpy import random

class Layer():
    """
    Superclass for all layers. Used for defining which methods must be implemented when implementing a new layer.
    """
    def __init_subclass__(cls, *a, **kw):
        for func in ['__init__', '_initialize', '_forward', '_backward']:
            if not func in dir(cls):
                raise NotImplementedError(f'{func} method not implemented for {cls.__name__} class. Please implement {func} method.')
            if not callable(getattr(cls, func)):
                raise TypeError(f'{func} is not callable for class {cls.__name__}. Please make sure {func} is a method.')
        super.__init_subclass__(*a, **kw)
    
    def __str__(self):
        ret_str = '{} layer with {} node(s) and {} as activation function'.format(self.__class__.__name__, self.nodes, self.activation_function)
        if self.bias:
            ret_str += ' and bias'
        return ret_str

    def is_first():
        return self._previous_layer == None

    def is_last():
        return self._next_layer == None

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
        self._weight_updates = None

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


    def full_multiplication(error, inputs):
        return np.multiply(error, inputs)
        
    def _backward(self):
        """
        Backward pass
        """
        output_derivatives = self.activation_function.derivative(self._wx) # d_afunc / d_wx
        input_derivatives = self._inputs # d_wx / d_weights

        if self.is_last():
            np.apply_along_axis(full_multiplication, 1, )
            self._weight_updates = self._backprop_pass_on * output_derivatives * input_derivatives
            self._backprop_pass_on = np.sum(self._backprop_pass_on * output_derivatives) # d_ET / d_wx

        else:
            error_derivatives = self._next_layer._weights # d_error / d_output
            self._weight_updates = self._backprop_pass_on * error_derivatives * output_derivatives * input_derivatives
            self._backprop_pass_on = self._backprop_pass_on * error_derivatives * output_derivatives
            print(type(self._next_layer._weights), type(self._next_layer._derivatives))
            #etotal = np.matmul(self.next_layer._weights.T, self.next_layer._derivatives.T).T
            #print(self.next_layer._weights)
            #print(self.next_layer._derivatives)
            #print(etotal)
