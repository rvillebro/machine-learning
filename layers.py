#!/bin/python
import activation
import numpy as np
from numpy import random
class Layer():
    def __init__(self):
        pass

class Dense(Layer):
    __slots__ = ['nodes', 'wx', 'weights', 'outputs', 'gradients', 'previous_layer', 'next_layer', 'activation_function']

    def __init__(self, nodes: int, activation_function = afunc.ReLU()):
        self.nodes = nodes
        self.outputs = np.zeros(nodes)
        self.previous_layer = None
        self.next_layer = None
        self.activation_function = activation_function
    
    def _initialize(self):
        if self.previous_layer is not None:
            self.weights = random.ranf((self.nodes, len(self.previous_layer.outputs) + 1))

    def _forward(self):
        previous_outputs = self.previous_layer.outputs

        # adds bias node
        x = np.append(previous_outputs, np.ones((previous_outputs.shape[0], 1)), axis= 1)

        # calculates outputs
        self.wx = np.matmul(x, self.weights.T)
        self.outputs = self.activation_function.fire(self.wx)
        return self.outputs

    def _backward(self):
        if self.next_layer is None:
            pl_outputs = self.previous_layer.outputs
            x = np.append(pl_outputs, np.ones((pl_outputs.shape[0], 1)), axis= 1)
            print(self.gradients.shape)
            print(self.activation_function.gradient(self.wx).shape)
            print(x.shape)
            print(self.weights.shape)
        else:
            pass 
        pass

    def get_node_info(self, index):
        return 'Node {}\nweights: {}\nbias: {}'.format(index, self.weights[index,:-1], self.weights[index,-1])