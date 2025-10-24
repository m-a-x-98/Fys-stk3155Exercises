import autograd.numpy as np

from Network import Network
from Activation import ReLU, Sigmoid, Softmax, Tanh
from Layers import Dense
from Loss import MSE, CrossEntropy
import Optimizers
from LearningRateScheduler import StepLR


from Metrics import ClassificationMetrics

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = (1 / (1 + 25 * np.power(X, 2))).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print(x_train.shape)
print(y_train.shape)



# Define the model
initLearningRate = 0.02
model = Network()

model.add(Dense((x_train.shape[1], 2), activation=Tanh()))
model.add(Dense((2, y_train.shape[1]), activation=Sigmoid()))

lrScheduler = StepLR(initLearningRate, 0.99, 100)

model.compile(MSE(), Optimizers.ADAM(initLearningRate, 0.9, 0.999))#, lrScheduler)

model.checkWithAutograd = False

# Train the model
epoch = 4000
for i in range(epoch):
    model.step(x_train, y_train)

    print(model.evaluate(x_test, y_test))

plt.scatter(x_test, y_test)
plt.scatter(x_test, model.predict(x_test))
plt.show()

