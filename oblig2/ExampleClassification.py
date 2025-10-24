import autograd.numpy as np

from Network import Network
from Activation import ReLU, Sigmoid, Softmax
from Layers import Dense
from Loss import MSE, CrossEntropy
import Optimizers
from LearningRateScheduler import StepLR


from Metrics import ClassificationMetrics

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# ----------------------------------------
# ----- Things that need to be fixed -----
# ----------------------------------------
#

#------ Things to add ---------
# Automatic picking test data from train data
# Examples of flowerdata and runge function in examples folder 


iris = datasets.load_iris()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

inputs = iris.data

# Since each prediction is a vector with a score for each of the three types of flowers,
# we need to make each target a vector with a 1 for the correct flower and a 0 for the others.
targets = np.zeros((len(iris.data), 3))
targets[np.arange(len(iris.target)), iris.target] = 1
#for i, t in enumerate(iris.target):
#    targets[i, t] = 1

def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris.data, targets, test_size = 0.2)
print(x_train.shape)
print(y_train.shape)


print("Done loading data")



# Define the model
initLearningRate = 0.01
model = Network()

model.add(Dense((x_train.shape[1], 10), activation=Sigmoid()))
model.add(Dense((10, y_train.shape[1]), activation=Softmax()))

lrScheduler = StepLR(initLearningRate, 0.98, 25)

model.compile(CrossEntropy(), Optimizers.ADAM(initLearningRate, 0.99, 0.99), lrScheduler)

model.checkWithAutograd = False


# Get information about the model
model.summary()
print("="*60)
model.specifics()


# Train the model
epoch = 4000
for i in range(epoch):
    model.step(x_train, y_train)

    print(model.evaluate(x_test, y_test))



# To get div metrics - requires classes to be defined in model
classes = [1, 2, 3]
model.defineClasses(classes)
yClasses = np.array(classes)[np.argmax(y_test, axis=1)]
metrics = ClassificationMetrics(model, x_test, yClasses)
metrics.oneFunctionToShowThemAll(binary=False)


# Use the model 
print(model.predict(np.array([0.4, 0.3, 0.2, 0.3])))
print(model.predictClasses(np.array([0.3, 0.3, 0.3, 0.3]))) # requires classes to be defined in model