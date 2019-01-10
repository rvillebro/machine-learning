#!/usr/bin/env python3
from loss_functions import LossFunction, MSE
from layers import Layer
from utils import shuffle, batch

class Network():
    def __init__(self):
        raise NotImplementedError('__init__ function not implemented for {} class'.format(self.__class__.__name__))

    def add(self):
        raise NotImplementedError('add function not implemented for {} class'.format(self.__class__.__name__))

    def insert(self):
        raise NotImplementedError('insert function not implemented for {} class'.format(self.__class__.__name__))

    def remove(self):
        raise NotImplementedError('remove not implemented function for {} class'.format(self.__class__.__name__))

    def train(self):
        raise NotImplementedError('train not implemented function for {} class'.format(self.__class__.__name__))

    def __str__(self):
        ret_str = '{} with {} layer(s) and {} as loss function\n'.format(self.__class__.__name__, len(self.layers), self.loss_function)
        ret_str += '\n'.join([str(layer) for layer in self.layers])
        return ret_str

class NeuralNetwork(Network):
    def __init__(self, loss_function: LossFunction = MSE()):
        self.layers = list()
        self.loss_function = loss_function

    def train(self, train_x, train_y, test_data=None, batch_size = 10, epochs=1):
        for layer in self.layers:
            layer._initialize()
        
        # looping epochs
        for _ in range(epochs):
            shuffle(train_x, train_y)
            
            for x, y in batch(train_x, train_y, batch_size):
                # sets input layer outputs (defines inputs to the neural network)
                self.layers[0]._outputs = x
                for layer in self.layers[1:]:
                    outputs = layer._forward()
                    print(outputs)
                
                y = y.reshape(outputs.shape)
                loss = self.loss_function.result(y, outputs)
                print(loss)
                derivatives = self.loss_function.derivative(y, outputs)
                self.layers[-1]._derivatives = derivatives
                for layer in reversed(self.layers[1:]):
                    layer._backward()
                break

    #### LAYERS #####
    def add(self, layer: Layer):
        if not isinstance(layer, Layer):
            raise TypeError('Please only add subclasses of Layer')
        if self.layers:
            layer.previous_layer = self.layers[-1]
            self.layers[-1].next_layer = layer
        self.layers.append(layer)


    def insert(self, index, layer):
        if index > len(self.layers)-1 or index < -len(self.layers):
            raise IndexError('Index out of bounds: {}'.format(index))

        layer.next_layer = self.layers[index]
        self.layers[index].previous_layer = layer

        # sets the
        if index > 0 or abs(index) < len(index)+1:
            layer.previous_layer = self.layers[index-1]
            self.layers[index-1].next_layer = layer

        self.layers.insert(index, layer)
        
    
    def remove(self, layer):
        try:
            index = self.layers.index(layer)
        except ValueError:
            raise ValueError('{} not found in {} layers'.format(str(layer), __class__.__name__))

        if index < len(self.layers):
            self.layers[index - 1].next_layer = self.layers[index+1]
        if index > 0:
            self.layers[index + 1].previous_layer = self.layers[index-1]
        
        self.layers.remove(layer)
