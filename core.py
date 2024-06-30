from abc import abstractmethod,ABC
from typing import Any, Tuple, List, Dict, Optional
from numpy import ndarray as NDArray

import numpy as np
# ——————————————————————————————————————————————————Op—————————————————————————————————————————————————————————————————————————————————
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
# ——————————————————————————————————————————————————Value—————————————————————————————————————————————————————————————————————————————————

class Value:
    op: Optional[Op] # 节点对应的计算操作， Op是自定义的计算操作类
    inputs: List["Value"]
    cached_data: NDArray
    requires_grad: bool


    def realize_cached_data(self): # 进行计算得到节点对应的变量，存储在cached_data里
        # 如果cached_data已经计算过，直接返回
        if self.cached_data is not None:
            return self.cached_data
        # 如果是叶子节点，没有计算操作，直接返回数据
        if self.is_leaf():
            return self.cached_data
        # 否则，使用op对inputs进行计算
        input_data = [input_value.realize_cached_data() for input_value in self.inputs]
        self.cached_data = self.op.compute(input_data)
        return self.cached_data


    def is_leaf(self):
        # 如果没有输入，认为是叶子节点
        return len(self.inputs) == 0

    def _init(self, op: Optional[Op], inputs: List["Value"],*, num_outputs: int = 1,
    cached_data: NDArray = None, requires_grad: Optional[bool] = None):
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad
        self.num_outputs = num_outputs



    @classmethod
    def make_const(cls, data,*, requires_grad=False): # 建立一个用data生成的独立节点
        # 创建一个常量节点
        return cls(cached_data=data, requires_grad=requires_grad)
    
    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]): # 根据op生成节点
        return cls(op=op, inputs=inputs)
    
# ——————————————————————————————————————————————————Tensor—————————————————————————————————————————————————————————————————————————————————

class Tensor (Value):
    grad: "Tensor" 

    def __init__(self, array, *, dtype=None, requires_grad=True, **kwargs):
        
        self.requires_grad = requires_grad
        self.data = array
        self.dtype = dtype
        self.grad = None
        for key, value in kwargs.items():
            setattr(self, key, value)


    @staticmethod
    def from_numpy(numpy_array, dtype):
        return Tensor(numpy_array, dtype=dtype, requires_grad=True)
    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        tensor.realize_cached_data()
        return tensor
    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, Tensor):
            tensor_data = data
        else:
            tensor_data = data.realize_cached_data()
        tensor._init(None, [], # 将前置节点置空
        cached_data = tensor_data, requires_grad = requires_grad)
        return tensor
    @ property
    def data (self): #对cached_data进行封装
        return self.realize_cached_data()
    @ data.setter
    def data (self, value):
        self.cached_data = value
    @ property
    def shape (self):
        return self.cached_data.shape
    @ property
    def dtype (self):
        return self.dtype


    def backward (self, out_grad=None):
        if out_grad:
            out_grad = out_grad
        else:
            out_grad = Tensor(np.ones(self.shape))
        compute_gradient_of_variables(self, out_grad)



    def detach (self):
        return Tensor.make_const(self.realize_cached_data())
    


# ——————————————————————————————————————————————————TensorOp—————————————————————————————————————————————————————————————————————————————————

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
    
class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a @ b
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad @ node.inputs[1].T, node.inputs[0].T @ out_grad


# ——————————————————————————————————————————————————辅助函数—————————————————————————————————————————————————————————————————————————————————

def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {} # dict结构，用于存储partial adjoint
    node_to_output_grads_list[output_tensor] = [out_grad]
    # 这里所求的序列，应该是只对tensor的序列，也就是说对与一些常量值，不应该排序
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor]))) # 请自行实现拓扑排序函数
    for node in reverse_topo_order:
        node.grad = sum(node_to_output_grads_list[node]) # 求node的partial adjoint之和，存入属性grad
        if node.is_leaf():
            continue
        for i, grad in enumerate(node.op.gradient(node.grad, node)): # 计算node.inputs的partial adjoint
            j = node.inputs[i]
            if j not in node_to_output_grads_list:
                node_to_output_grads_list[j] = []
            node_to_output_grads_list[j].append(grad) # 将计算出的partial adjoint存入dict

# 拓扑排序函数
def find_topo_sort(output_tensors):
    topo_order = []
    tensor_dict = {}
    # 创建一个字典，记录每个tensor的输入
    while len(output_tensors) > 0:
        tensor = output_tensors.pop()
        tensor_dict[tensor] = []
        for i in tensor.inputs:
            # 我这里认为，每个tensor的输入除了tensor之外，还有常量，
            # 而常量必须是value类型的，常量不参议梯度计算。常量的requires_grad属性是False
            if i.requires_grad:
                if i not in output_tensors:
                    output_tensors.append(i)
                tensor_dict[tensor].append(i)

    # 拓扑排序
    tmp = []
    # 如果入度为 0 就加入队列
    for i in tensor_dict:
        if len(tensor_dict[i]) == 0:
            tmp.append(i)
    
    while len(tmp) > 0:
        node = tmp.pop(0)
        topo_order.append(node)
        # 更新与之相关的入度
        for i in tensor_dict:
            if node in tensor_dict[i]:
                tensor_dict[i].remove(node)
                if len(tensor_dict[i]) == 0:
                    tmp.append(i)

    return topo_order



    

