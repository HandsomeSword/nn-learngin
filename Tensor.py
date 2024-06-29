import typing
from typing import List, Dict
from Value import Value
from Op import Op
import numpy as np


class Tensor (Value):
    grad: "Tensor" 

    def __init__(self, array, *, dtype=None, requires_grad=True, **kwargs):
        
        super().__init__(**kwargs)
        self.requires_grad = requires_grad
        self.data = array
        self.dtype = dtype
        self.grad = None



    @staticmethod
    def from_numpy(numpy_array, dtype):
        return Tensor(numpy_array, dtype=dtype, requires_grad=False)
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