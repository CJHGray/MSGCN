import numpy as np
import scipy.sparse as sp
import torch


'''
先将所有由字符串表示的标签数组用set保存，set的重要特征就是元素没有重复，
因此表示成set后可以直接得到所有标签的总数，随后为每个标签分配一个编号，创建一个单位矩阵，
单位矩阵的每一行对应一个one-hot向量，也就是np.identity(len(classes))[i, :]，
再将每个数据对应的标签表示成的one-hot向量，类型为numpy数组
'''
def encode_onehot(labels):
    classes = set(labels) # set() 函数创建一个无序不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}  # 字典 key为label的值，value为矩阵的每一行
    # np.identity()函数创建方阵，返回主对角线元素为1，其余元素为0的数组
    # 有多少个标签，这个方阵就是几乘几
    # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列，方阵变成索引序列
    # 同时列出数据和数据下标，一般用在for循环中
    # for i,j in enumerate(('a','b','c')):
    # 　　print i,j
    # 输出结果为：
    # >>>0 a
    # >>>1,b
    # >>>2,c
    # key为label值是因为label的值就是0,1,2,3...   即i
    # value就是第0行第一行第二行...   即c
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # get函数得到字典key对应的value，value映射到标签，也就是每个onehot向量映射到对应标签label
    return labels_onehot
    # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
    #  map(lambda x: x ** 2, [1, 2, 3, 4, 5])
    #  output:[1, 4, 9, 16, 25]


def load_data(path="../data/dblp/", dataset="dblp"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 读取../data/cora/cora.content
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
    # 1:-1表示从位置1到位置-1之前，-1表示最后一个
    # 这是2维的，第一个:表示行，所有行
    features = normalize(features)
    # 这个不是邻接矩阵的，是特征矩阵的归一化
    features = torch.FloatTensor(np.array(features.todense()))
    # 将numpy中的矩阵转化为torch中的tensor可以采用两种方法，一个是.torch.from_numpy()，另外一种是torch.Floattensor()。

    labels = encode_onehot(idx_features_labels[:, -1])
    # 这里的label为onthot格式，如第一类代表[1,0,0,0,0,0,0]
    # content file的每一行的格式为 ： <paper_id> <word_attributes>+ <class_label>
    #    分别对应 0, 1:-1, -1
    # [:, -1]表示最后一个
    # feature为第二列到倒数第二列，labels为最后一列
    labels = torch.LongTensor(np.where(labels)[1])
    # 这里将one-hot label转回index
    # print之后tensor([4, 5, 3,  ..., 2, 0, 4])，就是对应的标签索引
    # 把每个点的标签onehot矩阵变为对应标签编号



    # build graph_a
    # cites file的每一行格式为：  <cited paper ID>  <citing paper ID>
    # 根据前面的contents与这里的cites创建图，算出edges矩阵与adj 矩阵
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # content中的paper_ids，就是第一列
    idx_map = {j: i for i, j in enumerate(idx)}
    # 这里的j:i是idx_map中的存储形式{31336: 0, 1061127: 1, 1106406: 2, 13195: 3, 37879: 4。。。。。。}
    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
    # 哈希表中每一项为id: number，即节点id对应的编号为number
    # i为id序号，j为number，是对应的paper_id（string）
    edges_unordered_a = np.genfromtxt("{}{}_meta_1.cites".format(path, dataset),
                                    dtype=np.int32)
    # 读取了../data/cora/cora.cites
    # edges_unordered为直接从边表文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
    edges_a = np.array(list(map(idx_map.get, edges_unordered_a.flatten())),
                     dtype=np.int32).reshape(edges_unordered_a.shape)
    # 文件中两部分是paper_id，现在改成了映射过的索引id。样本之间的关系变为索引之间的关系
    # 边的edges_unordered中存储的是端点id，要将每一项的id换成编号。
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unordered形状一样的数组
    # [ 163  402]
    # [ 163  659]。。。这种形式，至少数字不会太大
    # reshape() Returns an array containing the same data with a new shape.
    adj_a = sp.coo_matrix((np.ones(edges_a.shape[0]), (edges_a[:, 0], edges_a[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # print(adj_a)
    # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # 所以先创建一个长度为edge_num的全1数组，每个1的填充位置就是一条边中两个端点的编号，
    # 即edges[:, 0], edges[:, 1]，矩阵的形状shape为(node_size, node_size)。
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    #
    # 经常的用法大概是这样的：coo_matrix((data, (i, j)), [shape=(M, N)])
    # 这里有三个参数：
    # data[:] 就是原始矩阵中的数据；edges.shape[0]就是边数，这里有5429个
    # edges.shape是(5429, 2)
    # i[:] 就是行的指示符号；labels.shape[0]就是节点数，这里有2708个
    # j[:] 就是列的指示符号；
    # labels.shape是(2708, 7) 因为每个点的标签都是长度为7的onehot，索引labels矩阵的形状为2708*7
    # shape参数是告诉coo_matrix原始矩阵的形状
    dense_a = sparse_mx_to_torch_sparse_tensor(adj_a).to_dense()
    # print(dense_a)
    # 没有规范化，并转换为稠密形式，方便计算度矩阵
    degree_a = torch.sum(dense_a, dim=0) + torch.sum(dense_a, dim=1)
    degree_a = degree_a.repeat(4, 1)
    degree_a = degree_a.permute(1, 0)


    # build symmetric adjacency matrix
    adj_a = adj_a + adj_a.T.multiply(adj_a.T > adj_a) - adj_a.multiply(adj_a.T > adj_a)
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，
    # 转换成无向图的邻接矩阵需要扩充成对称矩阵
    # 注意这里是无向图


    adj_a = normalize(adj_a + sp.eye(adj_a.shape[0]))
    # eye创建单位矩阵，第一个参数为行数，第二个为列数
    # 之前的邻接矩阵里对角上都是0
    # 对应公式A~=A+IN
    adj_a = sparse_mx_to_torch_sparse_tensor(adj_a)  # 邻接矩阵转为tensor处理
    # print(adj)这是已经归一化的对称邻接矩阵，并且做了转换处理，三元组
    # tensor(indices=tensor([[   0,    8,   14,  ..., 1389, 2344, 2707],
    #                        [   0,    0,    0,  ..., 2707, 2707, 2707]]),
    #        values=tensor([0.1667, 0.1667, 0.0500,  ..., 0.2000, 0.5000, 0.2500]),
    #        size=(2708, 2708), nnz=13264, layout=torch.sparse_coo)










    # build graph_b
    edges_unordered_b = np.genfromtxt("{}{}_meta_2.cites".format(path, dataset),
                                    dtype=np.int32)
    edges_b = np.array(list(map(idx_map.get, edges_unordered_b.flatten())),
                     dtype=np.int32).reshape(edges_unordered_b.shape)

    adj_b = sp.coo_matrix((np.ones(edges_b.shape[0]), (edges_b[:, 0], edges_b[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    dense_b = sparse_mx_to_torch_sparse_tensor(adj_b).to_dense()
    degree_b = torch.sum(dense_b, dim=0)
    degree_b = degree_b.repeat(4, 1)
    degree_b = degree_b.permute(1, 0)

    # build symmetric adjacency matrix
    adj_b = adj_b + adj_b.T.multiply(adj_b.T > adj_b) - adj_b.multiply(adj_b.T > adj_b)

    adj_b = normalize(adj_b + sp.eye(adj_b.shape[0]))
    adj_b = sparse_mx_to_torch_sparse_tensor(adj_b)  # 邻接矩阵转为tensor处理




    # build graph_c
    edges_unordered_c = np.genfromtxt("{}{}_meta_3.cites".format(path, dataset),
                                    dtype=np.int32)
    edges_c = np.array(list(map(idx_map.get, edges_unordered_c.flatten())),
                     dtype=np.int32).reshape(edges_unordered_c.shape)

    adj_c = sp.coo_matrix((np.ones(edges_c.shape[0]), (edges_c[:, 0], edges_c[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    dense_c = sparse_mx_to_torch_sparse_tensor(adj_c).to_dense()
    degree_c = torch.sum(dense_c, dim=0)
    degree_c = degree_c.repeat(4, 1)
    degree_c = degree_c.permute(1, 0)

    # build symmetric adjacency matrix
    adj_c = adj_c + adj_c.T.multiply(adj_c.T > adj_c) - adj_c.multiply(adj_c.T > adj_c)

    adj_c = normalize(adj_c + sp.eye(adj_c.shape[0]))
    adj_c = sparse_mx_to_torch_sparse_tensor(adj_c)  # 邻接矩阵转为tensor处理






    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    idx_train = range(140)  # 就是简单的0到139
    # print(idx_train)结果为range(0, 140)
    idx_val = range(200, 500)
    idx_test = range(500, 1496)

    idx_train = torch.LongTensor(idx_train)
    # [0,1,2,3,......138,139]
    idx_val = torch.LongTensor(idx_val)
    # [200,201,202,.....,498,499]
    idx_test = torch.LongTensor(idx_test)


    return degree_a, degree_b, degree_c, adj_a, adj_b, adj_c, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten() # 求倒数
    r_inv[np.isinf(r_inv)] = 0.
    # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)
    # 构建对角元素为r_inv的对角矩阵，每个的值是变成倒数的
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def accuracy(output, labels):  # 准确率，输出的预测的标签和应该的标签对比
    preds = output.max(1)[1].type_as(labels)
    # 使用type_as(tesnor)将张量转换为给定类型的张量。
    # 这里是把输出的类型转换为labels的数据类型
    # pred的应该是概率最大的那个
    correct = preds.eq(labels).double()
    # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):  # 把一个sparse matrix转为torch稀疏张量
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    # pytorch中的tensor转化成numpy中的ndarray : numpy()
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # coo矩阵用三元组表示，行、列、非零元素值
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 行和列
    values = torch.from_numpy(sparse_mx.data)
    # 值
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    # indices表示索引，哪些位置有值
    # values表示对应的索引下值是多少
    # shape是这个稀疏tensor的形状size，几乘几的
    # 举例
    # >>> i = torch.LongTensor([[0, 1, 1],
    #                           [2, 0, 2]])
    # >>> v = torch.FloatTensor([3, 4, 5])
    # >>> torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense()
    #  0  0  3
    #  4  0  5
    # [torch.FloatTensor of size 2x3]
    # 这里i为indices，v是values，torch.Size([2,3]是shape

