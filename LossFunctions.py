import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def calculate_loss(self, y_true, y_pred):
        raise NotImplementedError("Subclasses must implement loss()")
    
class MeanSquaredError(LossFunction):
    def calculate_loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
class CrossEntropyLoss(LossFunction):
    def calculate_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))