import autograd.numpy as np


class MSE:
    def __call__(self, predict, target):
        return np.mean((predict - target) ** 2)

    def derivative(self, predict, target):
        return 2*(predict - target)/target.size
        #return 2*(predict - target)/target.shape[0]

class CrossEntropy:
    def __call__(self, predict, target):
        return np.sum(-target * np.log(predict))

    def derivative(self, predict, target):
        return predict - target