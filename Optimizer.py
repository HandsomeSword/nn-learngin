from abc import ABC, abstractmethod
from core import Tensor

# ------------------------------Optimizer--------------------------------
class Optimizer(ABC):
    def __init__(self, params):
        self.params = params
    @abstractmethod
    def step(self):
        pass
    def reset_grad(self):
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr
    def step(self):
        for i, param in enumerate(self.params):
            grad = Tensor(param.grad, dtype='float32').data
            param.data= param.data - grad * self.lr

# ------------------------------Scheduler--------------------------------
class Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    @abstractmethod
    def step(self):
        pass

    def get_lr(self):
        return self.optimizer.lr
    
    def set_lr(self, lr):
        self.optimizer.lr = lr
    
# 恒定学习率
class ConstantScheduler(Scheduler):
    def __init__(self, optimizer, lr):
        super().__init__(optimizer)
        self.lr = lr
        self.set_lr(self.lr)


    def step(self):
        self.set_lr(self.lr)

# 学习率衰减
class StepScheduler(Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch % self.step_size == 0:
            new_lr = self.get_lr() * self.gamma
            self.set_lr(new_lr)