from abc import abstractmethod,ABC
from typing import Any, Tuple
from Value import Value
from numpy import ndarray as NDArray
from Tensor import Tensor
class Op(ABC):
    @abstractmethod
    def compute(self,*args: Tuple["NDArray"]) -> NDArray:
    # 前向计算. 参数args是由NDArray组成的序列Tuple，输出计算的结果NDArray
        pass
    @abstractmethod
    def gradient(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
    # 后向求导. 计算每个输入变量对应的局部伴随值(partial adjoint)
    # 参数out_grad是输出变量对应的伴随值，node是计算操作所在的计算图节点
    # 为方便编程，输出总是一个序列Tuple
        pass


class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    

# 对应元素相加
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad
# 加常数
class AddScalar(TensorOp): 
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, a: NDArray):
        return a + self.scalar
    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,) #重载gradient函数的输出必须是Tuple