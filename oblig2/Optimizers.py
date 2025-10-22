import autograd.numpy as np 
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def setN(self, nW, nb):
        pass
    @abstractmethod
    def __call__(self, grads):
        pass
    @abstractmethod
    def __str__(self):
        pass

class ADAM(Optimizer):
    """
    Implmenets the ADAM optimizer
    """
    def __init__(self, learningRate : float, decay1 : float, decay2 : float) -> None:
        self.beta1 = decay1
        self.beta2 = decay2
        self.lr = learningRate
        self.epsilon = 1E-8
        self.t = 0

    def setN(self, nW : int, nb : int) -> None:
        self.mW = np.zeros(nW)
        self.vW = np.zeros(nW)
        self.mb = np.zeros(nb)
        self.vb = np.zeros(nb)

    def __call__(self, grads : tuple[np.array, np.array]) -> np.array:
        self.t += 1
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * grads[0]
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * (grads[0] ** 2)
        m_hatW = self.mW / (1 - self.beta1 ** self.t)
        v_hatW = self.vW / (1 - self.beta2 ** self.t)
        dW = self.lr * m_hatW / np.sqrt(v_hatW + self.epsilon)

        self.mb = self.beta1 * self.mb + (1 - self.beta1) * grads[1]
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * (grads[1] ** 2)
        m_hatb = self.mb / (1 - self.beta1 ** self.t)
        v_hatb = self.vb / (1 - self.beta2 ** self.t)
        db = self.lr * m_hatb / np.sqrt(v_hatb + self.epsilon)
        return (dW, db)
    
    def __str__(self) -> str: 
        return f"ADAM, current learningrate = {self.lr}, beta1 = {self.beta1}, beta2 = {self.beta2}"

class RMSprop(Optimizer):
    """
    Implmenets the RMSprop optimizer
    """
    def __init__(self, learningRate : float, decay : float) -> None:
        self.lr = learningRate
        self.decay = decay
        self.epsilon = 1E-8

    def setN(self, nW : int, nb : int) -> None:
        self.movingAverage2W = np.zeros(nW)
        self.movingAverage2b = np.zeros(nb)

    def __call__(self, grads : tuple[np.array, np.array]) -> np.array:
        self.movingAverage2W = self.decay*self.movingAverage2W + (1-self.decay) * np.pow(grads[0], 2)
        self.movingAverage2b = self.decay*self.movingAverage2b + (1-self.decay) * np.pow(grads[1], 2)
        dW = self.lr * grads[0] / np.sqrt(self.movingAverage2W + self.epsilon)
        db = self.lr * grads[1] / np.sqrt(self.movingAverage2b + self.epsilon)
        return (dW, db)
    
    def __str__(self) -> str: 
        return f"RMSprop, current learningrate = {self.lr}, decay = {self.decay}"

class ADAgrad(Optimizer):
    """
    Implements the ADAgrad optimizer
    """
    def __init__(self, learningRate : float) -> None:
        self.lr = learningRate
        self.epsilon = 1E-8

    def setN(self, nW : int, nb : int) -> None:
        self.gradSumW = np.zeros(nW)
        self.gradSumb = np.zeros(nb)
    
    def __call__(self, grads : tuple[np.array, np.array]) -> np.array:
        self.gradSumW = self.gradSumW + np.pow(grads[0], 2)
        self.gradSumb = self.gradSumb + np.pow(grads[1], 2)
        lrW = self.lr / (np.sqrt(self.gradSumW)+self.epsilon)
        lrb = self.lr / (np.sqrt(self.gradSumb)+self.epsilon)
        return (lrW*grads[0], lrb*grads[1])
    
    def __str__(self) -> str: 
        return f"ADAgrad, current learningrate = {self.lr}"

class Basic(Optimizer):
    """
    Simple gradient descent with constant learningrate. 
    """
    def __init__(self, learningRate : float) -> None:
        self.lr = learningRate

    def setN(self, nW : int, nb : int) -> None:
        pass

    def __call__(self, grads : tuple[np.array, np.array]) -> np.array:
        return (self.lr * grads[0], self.lr * grads[1])
    
    def __str__(self) -> str: 
        return f"No optimizer, current learningrate = {self.lr}"

class Momentum(Optimizer):
    """
    Implements gradient descent with momentom
    """
    def __init__(self, learningRate : float, momentum : float) -> None:
        self.lr = learningRate
        self.momentum = momentum

    def setN(self, nW : int, nb : int) -> None:
        self.changeW = np.zeros(nW)
        self.changeb = np.zeros(nb)
    
    def __call__(self, grads : tuple[np.array, np.array]) -> np.array:
        Wg = self.lr * grads[0] + self.momentum*self.changeW
        bg = self.lr * grads[1] + self.momentum*self.changeb
        self.changeW = Wg
        self.changeb = bg
        return grads
    
    def __str__(self) -> str: 
        return f"Momentum, current learningrate = {self.lr}, momentum = {self.momentum}"