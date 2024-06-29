from typing import List, Optional
from numpy import ndarray as NDArray
from Op import Op

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