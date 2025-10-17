#import numpy as np
import autograd.numpy as np
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Softmax:
    def __init__(self, crossEntropy : bool):
        self.crossEntropy = crossEntropy

    def __call__(self, z):
        """Compute softmax values for each set of scores in the rows of the matrix z.
        Used with batched input data."""
        e_z = np.exp(z - np.max(z, axis=0))
        return e_z / np.sum(e_z, axis=1)[:, np.newaxis]
    
    def derivative(self, z):
        if self.crossEntropy: 
            return 1
        else:
            return 0 # should be else
        
class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, z):
        return np.where(z > 0, z, 0)

    def derivative(self, z):
        return np.where(z > 0, 1, 0)
    
class Sigmoid:
    def __call__(self, z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z):
        return np.exp(z) / (1 + np.exp(z)) ** 2
    
class Dummy:
    def __call__(self, z):
        return z

    def derivative(self, z):
        return 1
