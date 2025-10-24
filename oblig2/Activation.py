import autograd.numpy as np
from abc import ABC, abstractmethod

class Acitvation(ABC):
    @abstractmethod
    def __call__(self, z):
        pass
    @abstractmethod
    def derivative(self, z):
        pass
    @abstractmethod
    def __str__(self):
        "-Activation function-"

class Softmax(Acitvation):
    def __init__(self, crossEntropy : bool = True):
        self.crossEntropy = crossEntropy

    def __call__(self, z):
        """Compute softmax values for each set of scores in the rows of the matrix z.
        Used with batched input data."""
        e_z = np.exp(z)# - np.max(z, axis=0))
        return e_z / np.sum(e_z, axis=1)[:, np.newaxis]
    
    def derivative(self, z):
        if self.crossEntropy:
            return 1
        else:
            return 0 # should be else
        
    def __str__(self):
        return "Softmax"
        
class ReLU(Acitvation):
    def __init__(self):
        pass
    
    def __call__(self, z):
        return np.where(z > 0, z, 0)

    def derivative(self, z):
        return np.where(z > 0, 1, 0)

    def __str__(self):
        return "ReLU"
    
class Sigmoid(Acitvation):
    def __call__(self, z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z):
        return np.exp(z) / (1 + np.exp(z)) ** 2

    def __str__(self):
        return "Sigmoid"
    
class Dummy(Acitvation):
    def __call__(self, z):
        return z

    def derivative(self, z):
        return 1
    
    def __str__(self):
        return "None"

class Tanh(Acitvation):
    def __call__(self, z):
        return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

    def derivative(self, z):
        return 4 / np.power((np.exp(-z) + np.exp(z)), 2)

    def __str__(self):
        return "Tanh"