"""
从各个元路径npz文件和特征txt文件中生成模型的输入
需要生成：复合邻接矩阵adj，特征矩阵X，标签Y；X和Y分train和test
X的size为：所有house * feature_size（其中house会在训练时分成batch * Nodes）
Y的size为：所有house * label
通过索引区分训练测试集，即train_index, test_index
size均为 (month-1) * house_per_month * 1
"""

import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path, month_len, house_size):
    # 保证每个月的房屋数目（house_size）相等，data_size = month_len * house_size
    print('Loading data...')

    idx_features_labels = np.genfromtxt("{}n_feature.txt".format(path), dtype=np.float32)

    feature_size = idx_features_labels.shape[1]     # 特征维度
    data_size = idx_features_labels.shape[0]        # 所有月的总房屋个数

    features = idx_features_labels[:, 0:feature_size-2]  # 特征，去掉挂牌价和成交价
    listprice = idx_features_labels[:, -3]              # 挂牌价（训练过程中暂时不用）
    labels = idx_features_labels[:, -2]                 # 成交价
    listprice = listprice[:, np.newaxis]
    labels = labels[:, np.newaxis]                      # 列向量转列向量矩阵

    # listprice的最后一维添加索引
    index = np.array(range(0, data_size)).T
    index = index[:, np.newaxis]
    listprice = np.hstack((listprice, index))

    print('feature size: ' + str(features.shape))
    print('label size: ' + str(labels.shape))
    print('listprice size: ' + str(listprice.shape))

    # build graph
    # 这里将原来的normalize和转tensor的操作合并进了函数npz2tensor中
    # 参数输入为(metapath_name, filepath)，返回值类型为tensor
    # adj = []
    # adj.append(npz2array("H_community_H", path))
    # adj.append(npz2array("H_postal_H", path))
    # adj.append(npz2array("H_fsa_H", path))
    #adj.append(npz2array("H_municipality_H", path))
    # adj.append(npz2array("H_Garge_C_H", path))
    # adj = np.array(adj)

    # 制作训练集和测试集的索引
    index = range(0, data_size)
    train_index = []
    test_index = []
    for i in range(month_len - 1):
        train_index.append(index[i*house_size: (i+1)*house_size])
        test_index.append(index[(i+1)*house_size: (i+2)*house_size])
    train_index = np.array(train_index)
    test_index = np.array(test_index)

    # np.save(path + 'adj.npy', adj)
    np.save(path + 'features.npy', features)
    np.save(path + 'labels.npy', labels)
    np.save(path + 'listprice.npy', listprice)
    np.save(path + 'train_index.npy', train_index)
    np.save(path + 'test_index.npy', test_index)

    return features, labels, listprice, train_index, test_index


def normalize(mx, diag_lambda):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx + diag_lambda * sp.diags(mx.diagonal())  # 对角增强
    return mx


def normalize_torch(adj, diag_lambda):
    rowsum = torch.sum(adj, dim=1)
    r_inv = torch.pow(rowsum, -1)
    r_inv = torch.flatten(r_inv)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    adj = r_mat_inv.mm(adj)
    adj = adj + diag_lambda * torch.diag(torch.diag(adj, 0))  # 对角增强
    return adj


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.已改为转np.array"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    return np.array(sparse_tensor.to_dense())


def npz2array(metapath, filepath):
    data = sp.load_npz(filepath + metapath + ".npz")
    data = normalize(data, diag_lambda=5)
    data = sparse_mx_to_torch_sparse_tensor(data)
    print(metapath+": "+str(data.shape))
    return data

