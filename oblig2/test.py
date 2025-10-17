import autograd.numpy as np

from Network import Network
from Activation import ReLU, Sigmoid
from Layers import Dense
from Loss import MSE
from Optimizers import Basic

model = Network()
model.addOptimizer(Basic(0.01))
model.addCost(MSE())

model.add(Dense((6, 4), activation=ReLU()))
model.add(Dense((4, 2), activation=Sigmoid()))

model.checkWithAutograd = False

x = np.random.rand(10, 6)
y = np.random.rand(10, 2)

for i in range(10000):
    model.step(x, y)

    print(model.evaluate(x, y))
