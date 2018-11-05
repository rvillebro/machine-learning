#!/bin/python
import loss

class NeuralNetwork():
    __slots__ = ['layers', 'loss_function']
    
    def __init__(self, loss_function = lfunc.MSE()):
        self.layers = list()
        self.loss_function = loss_function

    def train(self, train_data, test_data, batch_size = 10, epochs=1):
        for layer in self.layers:
            layer._initialize()

        
        # looping epochs
        for i in range(epochs):
            train_data.shuffle()
            
            for X, y in train_data.batch(batch_size):
                # sets input layer outputs (defines inputs)
                self.layers[0].outputs = X
                for layer in self.layers[1:]:
                    outputs = layer._forward()
                
                gradients = self.loss_function.gradient(y, outputs)
                self.layers[-1].gradients = gradients
                for layer in reversed(self.layers[1:]):
                    layer._backward()
                    print(layer)
                break

    #### LAYERS #####
    def add(self, layer):
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
        except:
            print('Error')

        if index < len(self.layers):
            self.layers[index - 1].next_layer = self.layers[index+1]
        if index > 0:
            self.layers[index + 1].previous_layer = self.layers[index-1]
        
        self.layers.remove(layer)