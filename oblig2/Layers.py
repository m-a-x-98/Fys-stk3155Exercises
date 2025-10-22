import autograd.numpy as np
from abc import ABC, abstractmethod

from Activation import Dummy

class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass
    @abstractmethod
    def update(self, dW, db):
        pass
    @abstractmethod
    def getActivation(self):
        pass
    @abstractmethod
    def activationDer(self):
        pass
    @abstractmethod
    def summary(self):
        pass
    @abstractmethod
    def __str__(self):
        pass

class Dense(Layer):
    def __init__(self, shape : tuple[int, int], activation=None):
        self.W = np.random.rand(shape[1], shape[0])
        self.b = np.random.rand(shape[1])
        
        if activation == None:
            self.activation = Dummy()
        else:
            self.activation = activation

    def forward(self, x):
        z = x @ self.W.T + self.b
        a = self.activation(z)
        return z, a

    def update(self, dW, db):
        self.W = self.W - dW
        self.b = self.b - db
    
    def getActivation(self):
        return self.activation

    def activationDer(self):
        return self.activation.derivative
    
    def summary(self) -> tuple[int, int]:
        return (self.W.size, self.b.size)
    
    def __str__(self) -> str:
        return f"Layer Dense with activation {self.activation} and wegiht shape {self.W.shape} and bias shape {self.b.shape}"