from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def activate(self, inputs):
        raise NotImplementedError("Subclasses must implement activate()")

    def forward(self, inputs):
        return self.activate(inputs)

class ReLU(ActivationFunction):
    def activate(self, inputs):
        return np.maximum(np.zeros_like(inputs), inputs)

class LeakyReLU(ActivationFunction):
    def activate(self, inputs):
        return np.maximum(0.01 * inputs, inputs)
    
class Sigmoid(ActivationFunction):
    def activate(self, inputs):
        return 1 / (1 + np.exp(-inputs))

class Tanh(ActivationFunction):
    def activate(self, inputs):
        return (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
    
class Softmax(ActivationFunction):
    def activate(self, inputs):
        # softmax over classes (axis=0 aka along the columns)
        shifted = inputs - np.max(inputs, axis=0, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=0, keepdims=True)

