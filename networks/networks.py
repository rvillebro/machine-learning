#!/usr/bin/env python3
"""
Networks (networks)
==========================

This module contains all available networks: :class:`NeuralNetworks`, bla and bla. All networks must be a subclass of :class:`Network`.
"""

from .loss_functions import LossFunction, MSE
from .layers import Layer
from .utils import shuffle, batch

class Network():
    """
    Superclass for all networks. Used for defining which methods must be implemented when implementing a new network.
    """
    def __init_subclass__(cls, *a, **kw):
        for func in ['__init__', 'add', 'insert', 'remove', 'train', 'predict']:
            if not func in dir(cls):
                raise NotImplementedError(f'{func} method not implemented for {cls.__name__} class. Please implement {func} method.')
            if not callable(getattr(cls, func)):
                raise TypeError(f'{func} is not callable for class {cls.__name__}. Please make sure {func} is a method.')
        super.__init_subclass__(*a, **kw)

    def __str__(self):
        ret_str = '{} with {} layer(s) and {} as loss function\n'.format(self.__class__.__name__, len(self._layers), self.loss_function)
        ret_str += '\n'.join([str(layer) for layer in self._layers])
        return ret_str

class NeuralNetwork(Network):
    def __init__(self, loss_function: LossFunction = MSE()):
        """
        :param loss_function: The loss function of the network
        :type loss_function: :class:`loss_functions.LossFunction` 
        """
        if not isinstance(loss_function, LossFunction):
            raise TypeError('type {} not valid for argument loss_function. Please parse a subclass of LossFunction.'.format(type(loss_function)))
        
        #: The loss function of the network
        self.loss_function = loss_function
        #: A list of the layers in the network
        self._layers = list()

    def train(self, train_x, train_y, test_data=None, batch_size = 10, epochs=1):
        """
        :param train_X: The training data set
        :param train_y: The training validation set
        :param test_data: test data
        :param int batch_size: The batch size before updating weights
        :param int epochs: The number of epochs to train
        """
        for layer in self._layers:
            layer._initialize()
        
        # looping epochs
        for _ in range(epochs):
            shuffle(train_x, train_y)
            
            for x, y in batch(train_x, train_y, batch_size):
                # sets input layer outputs (defines inputs to the neural network)
                outputs = self.predict(x)
                
                y = y.reshape(outputs.shape)
                self.loss = self.loss_function.result(y, outputs)

                self._layers[-1]._backprop_pass_on = self.loss_function.derivative(y, outputs) # d_E_x / d_output_x (d_afunc)
                for layer in reversed(self._layers[1:]):
                    layer._backward()
                break

    def predict(self, x):
        """
        Give output of the network from the parsed data.

        :param x: The data for which predictions should be made
        :return: Prediction for parsed data x
        """
        self._layers[0]._outputs = x
        for layer in self._layers[1:]:
            layer._forward()

        return self._layers[-1]._outputs
        


    #### _layers #####
    def add(self, layer: Layer):
        if not isinstance(layer, Layer):
            raise TypeError('Please only add subclasses of Layer')
        if self._layers:
            layer._previous_layer = self._layers[-1]
            self._layers[-1]._next_layer = layer
        self._layers.append(layer)


    def insert(self, index, layer):
        if index > len(self._layers)-1 or index < -len(self._layers):
            raise IndexError('Index out of bounds: {}'.format(index))

        layer._next_layer = self._layers[index]
        self._layers[index]._previous_layer = layer

        # sets the
        if index > 0 or abs(index) < len(index)+1:
            layer._previous_layer = self._layers[index-1]
            self._layers[index-1]._next_layer = layer

        self._layers.insert(index, layer)
        
    
    def remove(self, layer):
        try:
            index = self._layers.index(layer)
        except ValueError:
            raise ValueError('{} not found in {} _layers'.format(str(layer), __class__.__name__))

        if index < len(self._layers):
            self._layers[index - 1]._next_layer = self._layers[index+1]
        if index > 0:
            self._layers[index + 1]._previous_layer = self._layers[index-1]
        
        self._layers.remove(layer)
