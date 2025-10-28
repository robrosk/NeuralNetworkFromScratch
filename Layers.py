from ActivationFunctions import ActivationFunction
import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError("Subclasses must implement forward()")

class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation_function):
        self.weights = 0.01 * np.random.randn(output_size, input_size)
        # R in output_size x input_size
        self.biases = np.zeros((output_size, 1))
        self.activation_function = activation_function
        
    def forward(self, inputs):
        return self.activation_function.forward(np.dot(self.weights, inputs) + self.biases)
        