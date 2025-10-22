import autograd.numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def __call__(self, predict, target):
        pass
    @abstractmethod
    def derivative(self, predict, target):
        pass
    @abstractmethod
    def __str__(self):
        pass

class MSE(Loss):
    def __call__(self, predict, target):
        return np.mean((predict - target) ** 2)

    def derivative(self, predict, target):
        return 2*(predict - target)/target.size
        #return 2*(predict - target)/target.shape[0]
    
    def __str__(self):
        return "MSE"

class CrossEntropy(Loss):
    def __call__(self, predict, target):
        return np.mean(-target * np.log(predict))

    def derivative(self, predict, target):
        return (predict - target) / (predict.size)
    
    def __str__(self):
        return "Cross entropy"