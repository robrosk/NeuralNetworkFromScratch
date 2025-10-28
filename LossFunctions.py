import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError("Subclasses must implement loss()")
    
    def forward(self, y_true, y_pred):
        return self.loss(y_true, y_pred)
    
class MeanSquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
class CrossEntropyLoss(LossFunction):
    def loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))