import autograd.numpy as np

from Activation import Dummy


class Dense:
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