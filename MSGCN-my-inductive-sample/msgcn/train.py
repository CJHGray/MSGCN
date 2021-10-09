from __future__ import division
from __future__ import print_function

import time
import argparse
# argparse 是python自带的命令行参数解析包，可以用来方便地读取命令行参数
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from msgcn.utils import load_data, accuracy
from msgcn.models import GCN

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# 随机种子seed起到固定初始值的作用
# 每个随机数都对应一个seed，固定了seed就是固定了随机数
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
# epochs
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learnig rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
# 为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
degree_a, degree_b, degree_c, adj_a, adj_b, adj_c, features, labels, idx_train, idx_val, idx_test, sample_adj, sample, idx_all = load_data()
# 归一化的对称邻接矩阵(稀疏表示)，归一化的特征，标签编号
print(sample)
print(sample.shape)


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
# 第一个参数是输入的特征向量的长度，第二个是隐层长度默认16，第三个是多少类，也就是有7个class，dropout默认为0.5
# 假如说输入特征是100维，gcn1使得100到16，gcn2使得16到7，然后根据输出判断是哪个class
# 这里就是对GCN模型的初始化Init，然后再GCN的init里对两个GraphConvolution也init，这样就有了两个W
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
# 这两个是一开始的参数，初始学习率，权重衰减(参数L2损失)

if args.cuda:
    model.cuda()
    features = features.cuda()
    degree_a = degree_a.cuda()
    degree_b = degree_b.cuda()
    degree_c = degree_c.cuda()
    adj_a = adj_a.cuda()
    adj_b = adj_b.cuda()
    adj_c = adj_c.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sample_adj = sample_adj.cuda()


def train(epoch):
    t = time.time()
    # 返回当前时间
    model.train()
    # 如果模型中有BN层(Batch Normalization）和Dropout，
    # 需要在训练时添加model.train()，在测试时添加model.eval()。
    # 其中model.train()是保证BN层用每一批数据的均值和方差，
    # 而model.eval()是保证BN用全部训练数据的均值和方差；
    # 而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
    optimizer.zero_grad()
    # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
    # pytorch中每一轮batch需要设置optimizer.zero_gra
    output_a = model(features, adj_a)
    # pytorch中两个tensor相乘就是哈达玛积
    output_a = output_a * degree_a
    output_b = model(features, adj_b)
    output_b = output_b * degree_b
    output_c = model(features, adj_c)
    output_c = output_c * degree_c
    output = output_a + output_b + output_c
    output = F.log_softmax(output, dim=1)

    # 这里直接调用GCN中的forward函数
    # 输入归一化的特征，归一化对称邻接矩阵
    # 返回的是logsoftmax之后的矩阵？
    loss_train = F.nll_loss(output[idx_all], labels[idx_all])
    # The negative log likelihood loss
    # NLLLoss的计算方式就是将上面output的值与对应的Label中的类别拿出来,去掉负号求均值。基于labels中的值取output的值

    # 由于在算output时已经使用了log_softmax，这里使用的损失函数就是NLLloss，如果前面没有加log运算，
    # 这里就要使用CrossEntropyLoss了
    # 损失函数NLLLoss() 的输入是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率，
    # 适合最后一层是log_softmax()的网络. 损失函数 CrossEntropyLoss() 与 NLLLoss() 类似,
    # 唯一的不同是它为我们去做 softmax.可以理解为：CrossEntropyLoss()=log_softmax() + NLLLoss()
    #
    # 如果想要损失最小，那就是对应的softmax值为1，也就是logsoftmax为0，全都接近0的话损失自然会小

    acc_train = accuracy(output[idx_all], labels[idx_all])
    # 计算准确率，这个是自己定义的函数在utils
    # 计算概率最大的标签是哪个，看有多少给和给出的label是一样的，计算比例
    loss_train.backward()
    # 反向求导  Back Propagation，计算当前梯度
    # backward只能被应用在一个标量上，也就是一个一维tensor，或者传入跟变量相关的梯度。
    optimizer.step()
    # 更新所有的参数  Gradient Descent，根据梯度更新当前网络
    # 首先需要明确optimzier优化器的作用, 形象地来说，
    # 优化器就是需要根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用


    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        # eval() 函数用来执行一个字符串表达式，并返回表达式的值
        output_a = model(features, adj_a)
        output_a = output_a * degree_a
        output_b = model(features, adj_b)
        output_b = output_b * degree_b
        output_c = model(features, adj_c)
        output_c = output_c * degree_c
        output = output_a + output_b + output_c
        output = F.log_softmax(output, dim=1)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # 验证集的损失函数
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # nllloss和accuracy同上
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

# 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
# 对训练好的模型做test
def test():
    model.eval()
    output_a = model(features, adj_a)
    # print(output_a)
    output_a = output_a * degree_a
    output_b = model(features, adj_b)
    # print(output_b)
    output_b = output_b * degree_b
    output_c = model(features, adj_c)
    # print(output_c)
    output_c = output_c * degree_c

    output1 = (output_a + output_b + output_c) / 3
    output = output1

    '''
    for hang in range(0, output1.shape[0]):
        output[hang] = (output1[sample_adj[hang][0]] + output1[sample_adj[hang][1]] + output1[sample_adj[hang][2]] +
                        output1[sample_adj[hang][3]] + output1[sample_adj[hang][4]] + output1[sample_adj[hang][5]] +
                        output1[sample_adj[hang][6]] + output1[sample_adj[hang][7]] + output1[sample_adj[hang][8]] +
                        output1[sample_adj[hang][9]])/10
                        '''


    for hang in range(0, output1.shape[0]):
        output[hang] = (output1[sample_adj[hang][0]] * sample[0][sample_adj[hang][0]] +
                        output1[sample_adj[hang][1]] * sample[0][sample_adj[hang][1]] +
                        output1[sample_adj[hang][2]] * sample[0][sample_adj[hang][2]] +
                        output1[sample_adj[hang][3]] * sample[0][sample_adj[hang][3]] +
                        output1[sample_adj[hang][4]] * sample[0][sample_adj[hang][4]] +
                        output1[sample_adj[hang][5]] * sample[0][sample_adj[hang][5]] +
                        output1[sample_adj[hang][6]] * sample[0][sample_adj[hang][6]] +
                        output1[sample_adj[hang][7]] * sample[0][sample_adj[hang][7]] +
                        output1[sample_adj[hang][8]] * sample[0][sample_adj[hang][8]] +
                        output1[sample_adj[hang][9]] * sample[0][sample_adj[hang][9]] )/\
                       (sample[0][sample_adj[hang][0]]+sample[0][sample_adj[hang][1]]+sample[0][sample_adj[hang][2]]+sample[0][sample_adj[hang][3]]+sample[0][sample_adj[hang][4]]+sample[0][sample_adj[hang][5]]+sample[0][sample_adj[hang][6]]+sample[0][sample_adj[hang][7]]+sample[0][sample_adj[hang][8]]+sample[0][sample_adj[hang][9]])

    # output = output_a + output_b + output_c
    # output = F.log_softmax(output, dim=1)
    # print(output)
    loss_test = F.nll_loss(output[idx_all], labels[idx_all])
    acc_test = accuracy(output[idx_all], labels[idx_all])
    # nllloss和accuracy同上
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    output = output.detach().numpy()

    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(output)
    X_pca = PCA(n_components=2).fit_transform(output)

    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(5, 5))
    # plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    plt.legend()
    plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()

# Train model
# Train model  逐个epoch进行train，最后test
t_total = time.time()
for epoch in range(args.epochs):
    # epoch 1-200
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
