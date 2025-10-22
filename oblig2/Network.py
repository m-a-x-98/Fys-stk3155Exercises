import autograd.numpy as np
from autograd import grad, elementwise_grad
import copy
from sklearn.utils import resample
from abc import ABC, abstractmethod

class NN(ABC):
    @abstractmethod
    def add(self, object):
        pass
    @abstractmethod
    def defineClasses(self, classes):
        pass
    @abstractmethod
    def compile(self, cost, optimizer, lrScheduler = None):
        pass
    @abstractmethod
    def forward(self, x):
        pass
    @abstractmethod
    def predict(self, x):
        pass
    @abstractmethod
    def step(self, input, target):
        pass
    @abstractmethod
    def evaluate(self, intput, target):
        pass
    @abstractmethod
    def summary(self):
        pass
    @abstractmethod
    def specifics(self):
        pass

class Network(NN):
    def __init__(self):
        self.layers = []
        self.checkWithAutograd = False
    
    def add(self, object):
        self.layers.append(object)

    def defineClasses(self, classes):
        self.classes = np.array(classes)

    def compile(self, cost, optimizer, lrScheduler = None):
        self.cost = cost
        self.lrScheduler = lrScheduler
        self.optimizers = []
        for i in range(len(self.layers)):
            opt = copy.deepcopy(optimizer)
            opt.setN(self.layers[i].W.shape[1], self.layers[i].b.shape[0])
            self.optimizers.append(opt)
        self.initLearningRate = optimizer.lr

    def forward(self, x):
        layerInputs = []
        zs = []
        a = x
        for layer in self.layers:
            layerInputs.append(a)
            z, a = layer.forward(a)

            zs.append(z)
        predict = a
        return layerInputs, zs, predict

    def predict(self, x):
        if (len(x.shape) == 1):
            x = x.reshape(1, -1)
        a = x
        for layer in self.layers:
            z, a = layer.forward(a)
        return a
    
    def predictClasses(self, x):
        a = self.predict(x)
        return self.classes[np.argmax(a, axis=1)]
    
    def _autoGradForward(self, input, layers, activation_funcs):
        a = input
        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W.T + b
            a = activation_func(z)
        return a

    def autoGradCost(self, layers, input, activation_funcs, target):
        predict = self._autoGradForward(input, layers, activation_funcs)
        return self.cost(predict, target)

    def backpropagation(self, input, target):
        layerInputs, zs, predict = self.forward(input)
        
        layer_grads = [() for layer in self.layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input, z, activationDer = layerInputs[i], zs[i], self.layers[i].activationDer()

            if i == len(self.layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost.derivative(predict, target)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                dC_da = dC_dz @ self.layers[i + 1].W

            dC_dz = dC_da * activationDer(z)
            dC_dW = dC_dz.T @ layer_input
            dC_db = np.sum(dC_dz, axis=0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads
    
    def step(self, input, target):
        x_train_re, y_train_re = resample(input, target)

        layer_grads = self.backpropagation(x_train_re, y_train_re)

        if self.checkWithAutograd: 
            cost_grad = grad(self.autoGradCost, 0)
            layers = []
            activationFuncs = []
            for l in self.layers:
                layers.append((l.W, l.b))
                activationFuncs.append(l.getActivation())
            dersAutoGrad = cost_grad(layers, x_train_re, activationFuncs, y_train_re)

        tol = 1E-13

        for i in range(len(layer_grads)):
            dW, db = self.optimizers[i](layer_grads[i])
            self.layers[i].update(dW, db)

            if self.checkWithAutograd:
                assert np.all(np.abs(layer_grads[i][0] - dersAutoGrad[i][0]) < tol)
                assert np.all(np.abs(layer_grads[i][1] - dersAutoGrad[i][1]) < tol)

        # Update learning rate 
        if self.lrScheduler != None:
            if self.lrScheduler.step():
                for optimizer in self.optimizers:
                    optimizer.lr = self.lrScheduler.update()


    def evaluate(self, input, target):
        _, _, pred = self.forward(input)
        return self.cost(pred, target)
    
    def summary(self):
        i = 0
        total = 0
        for layer in self.layers:
            paramW, paramb = layer.summary()
            print("-"*60)
            print(f"Layer {i} has {paramW + paramb} trainable parameters.")
            print("-"*60)
            total += paramW + paramb
            i += 1 

        print("-"*60)
        print(f"In total: ")
        print(f"{len(self.layers)} layers with {total} trainable parameters.")
        print("-"*60)

    def specifics(self):
        print(f"Optimizer: {self.optimizers[0]}.")
        print(f"Initial learning rate {self.initLearningRate}")
        print(f"Loss funciton: {self.cost}")
        for layer in self.layers:
            print(f"{layer}")
            print("-"*60)