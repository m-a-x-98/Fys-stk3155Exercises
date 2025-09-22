import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    PolynomialFeatures,
)  # use the fit_transform method of the created object!
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

n = 100
bootstraps = 1000

x = np.linspace(-3, 3, n)
y = np.exp(-(x**2)) + 1.5 * np.exp(-((x - 2) ** 2)) + np.random.normal(0, 0.1)

biases = []
variances = []
mses = []

def OLS_parameters(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

splitratio = 0.5

targets = np.ndarray((bootstraps, int(n*splitratio)))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = splitratio)
for b in range(bootstraps): targets[b, :] = y_test
for p in range(1, 10):
    predictions = np.ndarray((bootstraps, int(n*(1-splitratio))))

    X_train = normalize(x_train[:, None] ** np.arange(0, p+1), axis=0, norm='max')
    X_test = normalize(x_test[:, None] ** np.arange(0, p+1), axis=0, norm='max')
     
    for b in range(bootstraps):
        X_train_re, y_train_re = resample(X_train, y_train)

        # fit your model on the sampled data

        beta = OLS_parameters(X_train_re, y_train_re)

        # make predictions on the test data
        predictions[b, :] = X_test @ beta

    mse = np.average(np.pow(targets - predictions, 2))
    bias = np.average(np.pow(targets - np.average(predictions, axis=0), 2))
    variance = np.average(np.pow(predictions - np.average(predictions, axis=0), 2))
    biases.append(bias)
    variances.append(variance)
    mses.append(mse)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 18

plt.plot(range(1, 10), biases, label="Bias")
plt.plot(range(1, 10), variances, label="Viarance")
plt.plot(range(1, 10), mses, label="MSE")
plt.xlabel("Model complexity")
plt.ylabel("Error")
plt.legend()
plt.show()

def MSE(x, y):
    return np.sum((x - y)**2) / x.shape[0]

def getRidgeParam(x : np.array, y : np.array, l : int) -> np.array:
    return np.linalg.inv(l*np.identity(x.shape[1]) + x.T @ x) @ x.T @ y

def scale(xtrain, xtest, ytrian):
    scaler = StandardScaler()
    scaler.fit(xtrain)

    x_scaled_train = scaler.transform(xtrain)
    x_scaled_test = scaler.transform(xtest)
    y_offset = np.mean(ytrian)
    return x_scaled_train, x_scaled_test, y_offset

def polynomial_features(x : np.array, p : int, intercept : bool = True) -> np.array:
    return x[:, None] ** np.arange(int(intercept), p+1)

mse = []
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

xVals = range(1, 14)
lambdaVals = np.logspace(3, 5, 14, base=10)

for i in xVals:
    featMat = polynomial_features(x_train, i)
    featMatTest = polynomial_features(x_test, i)
    featMatScaled, featMatTestScaled, y_offset = scale(featMat, featMatTest, y_train)

    mseTemp = []
    for l in lambdaVals:
        beta = getRidgeParam(featMatScaled, y_train, l)
        mseTemp.append(MSE(featMatTestScaled@beta + y_offset, y_test))
    mse.append(mseTemp)


mse = np.array(mse)
plt.imshow(
    mse,
    aspect="auto",  # stretch so it fills
    extent=[lambdaVals[0], lambdaVals[-1], xVals[-1], xVals[0]],  # [xmin, xmax, ymin, ymax]
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 18

plt.xscale("log")
plt.xlabel("Lambda value")
plt.ylabel("Polynomial degree")
cbar = plt.colorbar()
cbar.set_label("MSE")
plt.show()
    