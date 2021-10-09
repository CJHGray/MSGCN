import torch.nn as nn
import torch.nn.functional as F
from msgcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        # 底层节点的参数，feature的个数，隐层节点个数，最终的分类数
        super(GCN, self).__init__()
        # super()._init_()在利用父类GCN里的对象构造函数

        self.gc1 = GraphConvolution(nfeat, nhid)
        # gc1输入尺寸nfeat，输出尺寸nhid
        # 这里会调用GraphConvolution中的init，这样就有了第一个W，1000多*16，还有bias
        self.gc2 = GraphConvolution(nhid, nclass)
        # gc2输入尺寸nhid，输出尺寸ncalss
        # # 这里会调用GraphConvolution中的init，这样就有了第一个W，16*7，还有bias
        self.dropout = dropout
        # 这里的dropout是个概率值，一般默认0.5
        # Dropout说的简单一点就是我们让在前向传播的时候，让某个神经元的激活值以一定的概率p让其失效（输出为0），失效相当从网络中移除

    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        # gc1(x, adj)这里调用了layer的forward函数
        # 假设点数为N，特征数为F
        # A即adj为N*N，x为N*F，W0为F*16
        # 得到的新的x为N*16
        x = F.dropout(x, self.dropout, training=self.training)
        # x是输入，self.dropout是每个元素随机被变为0的概率，默认0.5
        x = self.gc2(x, adj)
        # 调用layer的forward
        # 现在adj还是N*N，x是N*16，W1是16*7
        # 得到的新的x为N*7
        # return F.log_softmax(x, dim=1)
        return x
        # 理论上对于单标签多分类问题，直接经过softmax求出概率分布，
        # 然后把这个概率分布用crossentropy做一个似然估计误差。
        # 但是softmax求出来的概率分布，每一个概率都是(0,1)的，
        # 这就会导致有些概率过小，导致下溢。 考虑到这个概率分布总归是要经过crossentropy的，
        # 而crossentropy的计算是把概率分布外面套一个-log 来似然，
        # 那么直接在计算概率分布的时候加上log,把概率从（0，1）变为（-∞，0），
        # 这样就防止中间会有下溢出。
        # 所以log_softmax说白了就是将本来应该由crossentropy做的套log的工作提到预测概率分布来，
        # 跳过了中间的存储步骤，防止中间数值会有下溢出，使得数据更加稳定。
        # 正是由于把log这一步从计算误差提到前面，所以用log_softmax之后，
        # 下游的计算误差的function就应该变成NLLLoss(它没有套log这一步，
        # 直接将输入取反，然后计算和label的乘积求和平均)

        # 参数dim=1表示对每一行求softmax，那么每一行的值加起来都等于1。
        # LogSoftmax等于对Softmax求log值。
