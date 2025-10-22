from abc import ABC, abstractmethod

class LearningRateScheduler(ABC):
    @abstractmethod 
    def step(self):
        pass
    @abstractmethod 
    def update(self):
        pass


class StepLR(LearningRateScheduler):
    def __init__(self, initLearningRate : float, gamma : float, step_size : int) -> None: 
        self.gamma = gamma
        self.step_size = step_size
        self.lr = initLearningRate

        self.counter = 0
    
    def step(self) -> bool:
        self.counter += 1
        if (self.counter == self.step_size):
            self.counter = 0
            self.lr *= self.gamma
            return True
        return False
    
    def update(self) -> float:
        return self.lr