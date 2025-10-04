class Optimizers: 
    class ADAM:
        """
        Implmenets the ADAM optimizer
        """
        def __init__(self, learningRate, decay1, decay2, n):
            self.beta1 = decay1
            self.beta2 = decay2
            self.lr = learningRate
            self.m = np.zeros(n)
            self.v = np.zeros(n)
            self.epsilon = 1E-8
            self.t = 0

        def __call__(self, theta, gradient):
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            theta = theta - self.lr * m_hat / np.sqrt(v_hat + self.epsilon)
            return theta
        
        def __str__(self): return "ADAM   "

    class RMSprop:
        """
        Implmenets the RMSprop optimizer
        """
        def __init__(self, learningRate, decay, n):
            self.lr = learningRate
            self.decay = decay
            self.movingAverage2 = np.zeros(n)
            self.epsilon = 1E-8

        def __call__(self, theta, gradient):
            self.movingAverage2 = self.decay*self.movingAverage2 + (1-self.decay) * np.pow(gradient, 2)
            theta = theta - self.lr * gradient / np.sqrt(self.movingAverage2 + self.epsilon)
            return theta
        
        def __str__(self): return "RMSprop"

    class ADAgrad:
        """
        Implements the ADAgrad optimizer
        """
        def __init__(self, learningRate, n):
            self.learningRate = learningRate
            self.gradSum = np.zeros(n)
            self.epsilon = 1E-8

        def __call__(self, theta, gradient):
            self.gradSum = self.gradSum + np.pow(gradient, 2)
            lr = self.learningRate / (np.sqrt(self.gradSum)+self.epsilon)
            theta = theta - lr * gradient
            return theta
        
        def __str__(self): return "ADAgrad"

    class Simple:
        """
        Simple gradient descent with constant learningrate. 
        """
        def __init__(self, learningRate):
            self.learningRate = learningRate

        def __call__(self, theta, gradient):
            return theta - self.learningRate * gradient
        def __str__(self): return "No optimizer"

    class Momentum:
        """
        Implements gradient descent with momentom
        """
        def __init__(self, learningRate, momentum):
            self.learningRate = learningRate
            self.momentum = momentum
            self.lastTheta = None
        
        def __call__(self, theta, gradient):
            self.lastTheta = theta
            return theta - self.learningRate * gradient + self.momentum*(theta - self.lastTheta)
        
        def __str__(self): return "Momentum"









class GradientDescent:
    """
    Keeps and updates the parameters optimized by gradient descent.
    """
    def __init__(self, n_features, seed = 0):
        self.n_features = n_features
        if seed != 0: np.random.seed(seed)

        # Initialize weights for gradient descent
        self.theta = np.random.rand(n_features)

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def setGradient(self, gradient):
        self.gradient = gradient

    def forward(self, x_train, y_train):
        gradient = self.gradient(self.theta, x_train, y_train)
        self.theta = self.optimizer(self.theta, gradient)
        return self.theta
    





np.random.seed(1)

x_train, x_test, y_train, y_test = generateData(100)
x_train = x_train.flatten(); x_test = x_test.flatten()

epoch = 100
learningRate = 0.05
minChange = min(learningRate/1000, 0.001)
noIntercept = False

n_featuresList = [2, 4, 6]

logs = []

for n_features in n_featuresList:
    print("Numb features: ", n_features)
    # Define the search space for this number of features 
    featuresWithIntercept = n_features+int(not noIntercept)
    gradients = [Gradients.OLS(), 
                 Gradients.Ridge(0.01), 
                 Gradients.Lasso(0.01)]
    optimizers = [Optimizers.Simple(learningRate),
                  Optimizers.ADAM(learningRate, 0.9, 0.999, featuresWithIntercept),
                  Optimizers.RMSprop(learningRate, 0.99, featuresWithIntercept),
                  Optimizers.ADAgrad(learningRate, featuresWithIntercept),
                  Optimizers.Momentum(learningRate, 0.5)]
    
    combinations = [(grad, opt) for grad in gradients for opt in optimizers]
    
    gd = GradientDescent(featuresWithIntercept)
    for comb in combinations:
        gd.setOptimizer(comb[1])
        gd.setGradient(comb[0])

        x_test_feat = featureMat(x_test, n_features, noIntercept=noIntercept)

        mses = np.zeros(epoch)
        R2s = np.zeros(epoch)
        thetas = []
        numDiffs = 10
        mseDiffs = np.ones(numDiffs)

        best = None
        gotBest = False

        # Gradient descent loop
        for t in range(epoch):
            theta = gd.forward(featureMat(x_train, n_features, noIntercept=noIntercept), y_train)

            thetas.append(theta)
            mses[t], R2s[t] = testFit(x_test_feat, y_test, theta)

            # Early stopping
            mseDiffs[t%numDiffs] = abs(mses[t]-mses[t-1])

            if (np.all(mseDiffs < minChange)):
                break
        logs.append(f"Opt: {str(comb[1])}      grad: {str(comb[0])}      epoch: {t:3}      mse: {mses[t]:.4f}")
        
        #plt.plot(range(epoch), mses, label="MSEs")
        #plt.xlabel("Epoch")
        #plt.ylabel("Error")
        #plt.legend()
        #plt.show()

with open("out.txt", "w") as outfile:

    outfile.write("\n".join(logs))