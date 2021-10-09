import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    # 初始化层：输入feature，输出feature，权重，偏移
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # 这里是创了一个矩阵，比如in_features是4*1的，out是3*1，那么就是一个3*4的矩阵
        # 3*4的矩阵乘4*1得到3*1
        # 常见用法self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))：
        # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter
        # 绑定到这个module里面，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

        # 选择Parameter类实例作为weight和bias的存储方式
        # PyTorch创建了一个权值矩阵，并使用随机值初始化它
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            # Parameters与register_parameter都会向parameters写入参数，但是后者可以支持字符串命名
        self.reset_parameters()
        # 初始化权重

    # 初始化权重
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # sqrt平方根
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        self.weight.data.uniform_(-stdv, stdv)
        # 使用均匀分布U(a,b)初始化Tensor，即Tensor的填充值是等概率的范围为 [a，b) 的值。均值为 （a + b）/ 2
        # 那这里就是等概率的-stdv和stdv之间的值，均值为0.
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    '''
        前馈运算 即计算A~ X W(0)
        input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
        直接输入与权重之间进行torch.mm操作，得到support，即XW
        support与adj进行torch.mm操作，得到output，即AXW选择是否加bias
    '''
    def forward(self, input, adj):
        # 就算没有哪里调用forward这里也会执行
        support = torch.mm(input, self.weight)
        # weight在前边
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        # 通过设置断点，可以看出output的形式是0.01，0.01，0.01，0.01，0.01，0.01，0.94]，
        # 里面的值代表该x对应标签不同的概率，故此值可转换为#[0,0,0,0,0,0,1]，
        # 对应我们之前把标签onthot后的第七种标签

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
