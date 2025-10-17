import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)


# Derivative of the ReLU function
def ReLU_der(z):
    return np.where(z > 0, 1, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mse(predict, target):
    return np.mean((predict - target) ** 2)


def mse_der(predict, target):
    return 2*(predict - target)/target.size


def sigmoid_der(z):
    return np.exp(z)/(1+np.exp(z))**2


def create_layers(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers


def feed_forward(input, layers, activation_funcs):
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W.T + b
        a = activation_func(z)
    return a


def feed_forward_saver(input, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W.T + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a


def cost(layers, input, activation_funcs, target):
    predict = feed_forward(input, layers, activation_funcs)
    return mse(predict, target)


def backpropagation(
    input, layers, activation_funcs, target, activation_ders, cost_der=mse_der
):
    layer_inputs, zs, predict = feed_forward_saver(input, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = cost_der(predict, target)
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W, b) = layers[i + 1]
            dC_da = dC_dz @ W

        dC_dz = dC_da * activation_der(z)
        dC_dW = dC_dz.T @ layer_input 
        dC_db = np.sum(dC_dz, axis=0)

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads


network_input_size = [20, 10]
layer_output_sizes = [10, 4, 3, 2]
activation_funcs = [sigmoid, ReLU, ReLU, ReLU]
activation_ders = [sigmoid_der, ReLU_der, ReLU_der, ReLU_der]

layers = create_layers(network_input_size[1], layer_output_sizes)

x = np.random.rand(*network_input_size)
target = np.random.rand(*[network_input_size[0], layer_output_sizes[-1]])

#print(feed_forward(x, layers, activation_funcs))

layer_grads = backpropagation(x, layers, activation_funcs, target, activation_ders)
(g_1, g_2, g_3, g_4) = layer_grads

cost_grad = grad(cost, 0)
(autoG1, autoG2, autoG3, autoG4) = cost_grad(layers, x, activation_funcs, target)
print(g_1[0]-autoG1[0])
print(g_2[0]-autoG2[0])
print(g_3[0]-autoG3[0])
print(g_4[0]-autoG4[0])
print(g_1[1]-autoG1[1])
print(g_2[1]-autoG2[1])
print(g_3[1]-autoG3[1])
print(g_4[1]-autoG4[1])