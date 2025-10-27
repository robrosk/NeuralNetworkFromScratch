from ActivationFunctions import ActivationFunction
import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = 0.01 * np.random.randn(output_size, input_size)
        self.biases = np.zeros((1, output_size))
        self.activation_function = activation_function
        
    def forward(self, inputs):
        return self.activation_function.forward(np.dot(self.weights, inputs) + self.biases)
        