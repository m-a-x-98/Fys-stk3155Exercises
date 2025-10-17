
class Basic:
    def __init__(self, learningRate):
        self.lr = learningRate

    def __call__(self, grads):
        (Wg, bg) = grads
        return (self.lr * Wg, self.lr * bg)