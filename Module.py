from abc import abstractmethod,ABC
from core import *
import numpy


# ------------------------------Module--------------------------------
class Module(ABC):
    def __init__(self):
        self.training = True

    def parameters(self) -> List["Tensor"]:
        return _unpack_params(self.__dict__)
    
    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    @abstractmethod
    def forward(self):
        pass
    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False
    def train(self):
        self.training = True
        for m in self._children():
            m.training = True


# ------------------------------Sub Module--------------------------------
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init_He (in_features, out_features, dtype) #请自行实现初始化算法if bias:
        if bias:
            self.bias = Parameter(numpy.zeros(self.out_features))
        else:
            self.bias = None
    def forward(self, X: Tensor) -> Tensor:
        X_out = X @ self.weight
        if self.bias:
            return X_out + self.bias.broadcast_to(X_out.shape)
        return X_out

class Sequential(Module):
    def __init__(self,*modules):
        super().__init__()
        self.modules = modules
    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

class Flatten(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)
    

class ReLU(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        x_out = x.exp()
        return x_out / (x_out + 1)

class Softmax(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        exps = x.exp()
        return exps / exps.sum()

class CrossEntrophyLoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TODO: 交叉熵损失
        return -y_true * y_pred.log()

class MSELoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return (y_pred - y_true).pow(2).sum()

class Residual(Module):
    def __init__(self, sublayer: Module):
        super().__init__()
        self.sublayer = sublayer
    def forward(self, x: Tensor) -> Tensor:
        return x + self.sublayer(x)

# ------------------------------Parameter--------------------------------
class Parameter(Tensor):
    def __init__(self, data, requires_grad=True, dtype=None):
        super().__init__(data, dtype=dtype,requires_grad=requires_grad, inputs = [], op = None)




# ------------------------------other function--------------------------------
def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter): return [value]
    elif isinstance(value, Module): return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else: return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


def init_He(in_features, out_features, dtype):
    return Parameter(numpy.random.randn(in_features, out_features) * numpy.sqrt(2 / in_features), dtype=dtype)