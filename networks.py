#!/usr/bin/env python3
from loss_functions import LossFunction, MSE
from layers import Layer
from utils import shuffle, batch

class Network():
    def __init__(self):
        raise NotImplementedError('__init__ method not implemented for {} class. Please implement __init__ method.'.format(self.__class__.__name__))

    def add(self):
        raise NotImplementedError('add method not implemented for {} class. Please implement add method.'.format(self.__class__.__name__))

    def insert(self):
        raise NotImplementedError('insert method not implemented for {} class. Please implement insert method.'.format(self.__class__.__name__))

    def remove(self):
        raise NotImplementedError('remove method not implemented for {} class. Please implement remove method.'.format(self.__class__.__name__))

    def train(self):
        raise NotImplementedError('train method not implemented for {} class. Please implement train method.'.format(self.__class__.__name__))
    
    def predict(self):
        raise NotImplementedError('predict method not implemented for {} class. Please implement predict method.'.format(self.__class__.__name__))

    def __str__(self):
        ret_str = '{} with {} layer(s) and {} as loss function\n'.format(self.__class__.__name__, len(self._layers), self.loss_function)
        ret_str += '\n'.join([str(layer) for layer in self._layers])
        return ret_str

class NeuralNetwork(Network):
    def __init__(self, loss_function: LossFunction = MSE()):
        if not isinstance(loss_function, LossFunction):
            raise TypeError('type {} not valid for argument loss_function. Please parse a subclass of LossFunction.'.format(type(loss_function)))
        
        self.loss_function = loss_function
        self._layers = list()

    def train(self, train_x, train_y, test_data=None, batch_size = 10, epochs=1):
        for layer in self._layers:
            layer._initialize()
        
        # looping epochs
        for _ in range(epochs):
            shuffle(train_x, train_y)
            
            for x, y in batch(train_x, train_y, batch_size):
                # sets input layer outputs (defines inputs to the neural network)
                outputs = self.predict(x)
                
                y = y.reshape(outputs.shape)
                loss = self.loss_function.result(y, outputs)
                derivatives = self.loss_function.derivative(y, outputs)
                self._layers[-1]._derivatives = derivatives
                for layer in reversed(self._layers[1:]):
                    layer._backward()
                break

    def predict(self, x):
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
