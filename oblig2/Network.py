import autograd.numpy as np
from autograd import grad, elementwise_grad

class Network:
    def __init__(self):
        self.layers = []
        self.checkWithAutograd = False

    def add(self, object):
        self.layers.append(object)

    def addCost(self, cost):
        self.cost = cost

    def addOptimizer(self, optimizer):
        self.optimizer = optimizer

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

    def autoGradCost(self, layers, input, activation_funcs, target):
        _, _, predict = self.forward(input)
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
        layer_grads = self.backpropagation(input, target)

        if self.checkWithAutograd: 
            cost_grad = grad(self.autoGradCost, 0)
            layers = []
            activationFuncs = []
            for l in self.layers:
                layers.append((l.W, l.b))
                activationFuncs.append(l.getActivation())
            dersAutoGrad = cost_grad(layers, input, activationFuncs, target)

        tol = 1E-8

        for i in range(len(layer_grads)):
            dW, db = self.optimizer(layer_grads[i])
            self.layers[i].update(dW, db)

            if self.checkWithAutograd:
                print(dW)
                print(dersAutoGrad)
                assert np.all(np.abs(dW - dersAutoGrad[i][0]) < tol)
                assert np.all(np.abs(db - dersAutoGrad[i][1]) < tol)

    def evaluate(self, input, target):
        _, _, pred = self.forward(input)
        return self.cost(pred, target)