import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
"""
GCN层：接受全部HIN邻接矩阵(meta_size * Nodes * Nodes)，
       以及全部特征矩阵X（Nodes * input_dim)
将每个月内的Nodes个LSTM embedding都转成graph embedding
输出为全部数据的graph embedding：Nodes * output_dim
"""


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # adj = adj + torch.eye(adj.shape[0],adj.shape[0]).cuda()  # A+I
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN2lv(nn.Module):
    def __init__(self, nfeat, gc1_outdim, gc2_outdim, dropout, meta_size):
        super(GCN2lv, self).__init__()
        self.meta_size = meta_size
        self.gc1_outdim = gc1_outdim
        self.gc2_outdim = gc2_outdim
        # 用Variable引入权重
        self.W = Parameter(torch.FloatTensor(self.meta_size, 1))
        nn.init.xavier_uniform_(self.W.data)

        self.gc1 = GraphConvolution(nfeat, gc1_outdim)
        self.gc2 = GraphConvolution(gc1_outdim, gc2_outdim)
        self.dropout = dropout

    def forward(self, adj, x):
        # 每条meta-graph分别传入GCN
        gcn_out = []
        shape = x.shape[0]
        for i in range(self.meta_size):
            gcn_out.append(F.relu(self.gc1(x, adj[i])))
            gcn_out[i] = F.relu(self.gc2(gcn_out[i], adj[i]))
            gcn_out[i] = gcn_out[i].view(1,shape*self.gc2_outdim)

        x = gcn_out[0]
        for i in range(1, self.meta_size):
            x = torch.cat((x, gcn_out[i]), 0)
        x = torch.t(x)
        # print(self.W)
        x = F.relu(torch.mm(x, self.W))
        x = x.view(shape, self.gc2_outdim)
        x = F.dropout(x,  self.dropout, training=self.training)
        return x


# 公用LSTM版
class r_gcn2lv_1LSTMs(nn.Module):
    def __init__(self, gcn_input_dim, gc1_out_dim, lstm_input_dim, hidden_dim,
                 label_out_dim, meta_size, all_month, month_len, layers=1, dropout=0.2):
        super(r_gcn2lv_1LSTMs, self).__init__()
        self.hidden_dim = hidden_dim
        self.meta_size = meta_size
        self.all_month = all_month
        self.month_len = month_len
        self.glstm_list = []
        for i in range(month_len):
            self.glstm_list.append(GCN2lv(nfeat=gcn_input_dim, gc1_outdim=gc1_out_dim, gc2_outdim=lstm_input_dim,
                                          dropout=dropout, meta_size=meta_size))
        self.glstm = nn.ModuleList(self.glstm_list)
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.hidden_dim, num_layers=layers)
        # self.linear_gcn = nn.Linear(hidden_dim, gcn_input_dim)  # 暂时输入输入维度一致，后续可再调整
        self.linear_price = nn.Linear(gcn_input_dim, label_out_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, adj, x, y_index):
        """
        :param x: Nodes * input_dim
        :param adj: meta_size * Nodes * Nodes
        :param y_index: last_month(1) * batch_size
        :return: 全局的features，以及最后一个月的房价
        """
        Nodes, num_features = x.size()
        month_len, batch_size = y_index.size()
        # print('y_index: ' + str(y_index.shape))
        # print('month_len: '+ str(month_len))
        house_size = int(Nodes / self.all_month)
        out_allmonth = x
        for i in range(0, self.month_len):
            g_emb = self.glstm[i](adj, out_allmonth)  # Nodes * lstm_input_dim
            # print('g_emb: ' + str(g_emb.shape))
            # 将g_emb分割成全长的时间序列
            seq_list = []
            for i in range(self.all_month):
                seq_list.append(
                    g_emb.index_select(0, torch.LongTensor(range(i * house_size, (i + 1) * house_size)).cuda()))  # 0按行，1按列
            sequence = torch.stack(seq_list, 0)  # month_len, batch_size, lstm_input_dim
            # print('sequence: ' + str(sequence.shape))
            # 将GCN生成的全部嵌入放入LSTM训练
            out, hidden = self.lstm(sequence)  # out:(month_len, house_size, hidden_size)
            # print('out: ' + str(out.shape))
            out_allmonth_t = out.view(Nodes, self.hidden_dim)  # 全部数据经过LSTM后的嵌入  Nodes*LSTM_hidden_size
            # print('out_allmonth_t: ' + str(out_allmonth_t.shape))
            # out_allmonth = self.linear_gcn(out_allmonth_t)  # 输出1：全部房子的embedding
            out_price_t = self.linear_price(out_allmonth_t)
            # 将发生交易的房子的标签取出来，用作反向传播的信号
            # 完全取决于y_index长度，y_index包含哪几个月，就取哪几个月的label
            label_list = []
            for i in range(month_len):
                label_list.append(out_price_t.index_select(0, y_index[i]))
            out_price = torch.stack(label_list, 0)  # 输出2：参加本月交易的房子的label
        return out_allmonth, self.LeakyReLU(out_price)


class GCN2lv_static(nn.Module):
    def __init__(self, nfeat, gc1_outdim, gc2_outdim, dropout, meta_size):
        super(GCN2lv_static, self).__init__()
        self.meta_size = meta_size
        self.gc1_outdim = gc1_outdim
        self.gc2_outdim = gc2_outdim
        # 用Parameter引入权重
        self.W = Parameter(torch.FloatTensor(self.meta_size, 1))
        nn.init.xavier_uniform_(self.W.data)
        self.gc1 = GraphConvolution(nfeat, gc1_outdim)
        self.gc2 = GraphConvolution(gc1_outdim, gc2_outdim)
        self.dropout = dropout
        self.dense2 = nn.Linear(gc2_outdim, 1)

    def forward(self, adj, x):
        # 每条meta-graph分别传入GCN
        gcn_out = []
        shape = x.shape[0]
        for i in range(self.meta_size):
            gcn_out.append(F.relu(self.gc1(x, adj[i])))
            gcn_out[i] = F.relu(self.gc2(gcn_out[i], adj[i]))
            gcn_out[i] = gcn_out[i].view(1,shape*self.gc2_outdim)

        x = gcn_out[0]
        for i in range(1, self.meta_size):
            x = torch.cat((x, gcn_out[i]), 0)
        x = torch.t(x)
        x = F.relu(torch.mm(x, self.W))
        # print(self.W,flush=True)
        x = x.view(shape,self.gc2_outdim)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.dense2(x)
        return x


class GCNlstm_static(nn.Module):
    def __init__(self, nfeat, gc1_outdim, gc2_outdim, dropout, meta_size, house_size):
        super(GCNlstm_static, self).__init__()
        self.meta_size = meta_size
        self.gc1_outdim = gc1_outdim
        self.gc2_outdim = gc2_outdim
        self.house_size = house_size
        # 用Parameter引入权重
        self.W = Parameter(torch.FloatTensor(self.meta_size, 1))
        nn.init.xavier_uniform_(self.W.data)

        self.gc1 = GraphConvolution(nfeat, gc1_outdim)
        self.gc2 = GraphConvolution(gc1_outdim, gc2_outdim)
        self.lstm = nn.LSTM(input_size=gc2_outdim, hidden_size=gc2_outdim, num_layers=1)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = dropout
        self.linear_price = nn.Linear(gc2_outdim, 1)

    def forward(self, adj, x):
        # 每条meta-graph分别传入GCN
        gcn_out = []
        shape = x.shape[0]
        house_size = self.house_size
        seq_len = int(x.shape[0]/house_size)
        for i in range(self.meta_size):
            gcn_out.append(F.relu(self.gc1(x, adj[i])))
            gcn_out[i] = F.relu(self.gc2(gcn_out[i], adj[i]))
            gcn_out[i] = gcn_out[i].view(1, shape*self.gc2_outdim)

        x = gcn_out[0]
        for i in range(1, self.meta_size):
            x = torch.cat((x, gcn_out[i]), 0)
        x = torch.t(x)
        x = F.relu(torch.mm(x, self.W))
        # print(self.W,flush=True)
        x = x.view(shape, self.gc2_outdim)
        x = F.dropout(x, self.dropout, training=self.training)
        seq_list = []
        for i in range(seq_len):
            seq_list.append(
                x.index_select(0, torch.LongTensor(range(i * house_size, (i + 1) * house_size)).cuda()))  # 0按行，1按列
        sequence = torch.stack(seq_list, 0)  # month_len, batch_size, lstm_input_dim
        # print('sequence: ' + str(sequence.shape))
        # 将GCN生成的全部嵌入放入LSTM训练
        out, hidden = self.lstm(sequence)  # out:(month_len, batch_size, hidden_size)
        out = out.view(shape, self.gc2_outdim)
        x = self.linear_price(out)
        return x



#  定义T-GCN模型
class T_GCN(nn.Module):
    def __init__(self, nfeat, gc1_outdim, gc2_outdim, dropout, meta_size, house_size):
        super(T_GCN, self).__init__()
        self.meta_size = meta_size
        self.gc1_outdim = gc1_outdim
        self.gc2_outdim = gc2_outdim
        self.house_size = house_size
        # 用Parameter引入权重
        self.W = Parameter(torch.FloatTensor(self.meta_size, 1))
        nn.init.xavier_uniform_(self.W.data)
        self.gc1 = GraphConvolution(nfeat, gc1_outdim)
        self.gc2 = GraphConvolution(gc1_outdim, gc2_outdim)
        # 引入GRU
        # 这里的input_size就是词向量的维度，hidden_size是隐藏层的维度, n_layers是GRU的层数
        self.gru = nn.GRU(input_size=gc2_outdim, hidden_size=gc2_outdim, num_layers=1)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = dropout
        self.linear_price = nn.Linear(gc2_outdim, 1)

    def forward(self, adj, x):
        # 每条meta-graph分别传入GCN
        gcn_out = []
        shape = x.shape[0]
        house_size = self.house_size
        seq_len = int(x.shape[0]/house_size)
        for i in range(self.meta_size):
            gcn_out.append(F.relu(self.gc1(x, adj[i])))
            gcn_out[i] = F.relu(self.gc2(gcn_out[i], adj[i]))
            gcn_out[i] = gcn_out[i].view(1, shape*self.gc2_outdim)

        x = gcn_out[0]
        for i in range(1, self.meta_size):
            x = torch.cat((x, gcn_out[i]), 0)
        x = torch.t(x)
        x = F.relu(torch.mm(x, self.W))
        # print(self.W,flush=True)
        x = x.view(shape, self.gc2_outdim)
        x = F.dropout(x, self.dropout, training=self.training)

        seq_list = []
        for i in range(seq_len):
            seq_list.append(
                x.index_select(0, torch.LongTensor(range(i * house_size, (i + 1) * house_size)).cuda()))  # 0按行，1按列
        sequence = torch.stack(seq_list, 0)  # month_len, batch_size, lstm_input_dim
        # print('sequence: ' + str(sequence.shape))
        # 将GCN生成的全部嵌入放入LSTM训练
        out, hidden = self.gru(sequence)
        # out, hidden = self.lstm(sequence)  # out:(month_len, batch_size, hidden_size)
        out = out.view(shape, self.gc2_outdim)
        x = self.linear_price(out)
        return x



class LSTM_static(nn.Module):
    def __init__(self, nfeat, dropout,  house_size):
        super(LSTM_static, self).__init__()
        self.house_size = house_size
        self.nfeat = nfeat
        self.lstm = nn.LSTM(input_size=nfeat, hidden_size=nfeat, num_layers=1)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = dropout
        self.linear_price = nn.Linear(nfeat, 1)

    def forward(self, x):
        shape = x.shape[0]
        house_size = self.house_size
        seq_len = int(x.shape[0]/house_size)
        # 构造时间序列
        seq_list = []
        for i in range(seq_len):
            seq_list.append(
                x.index_select(0, torch.LongTensor(range(i * house_size, (i + 1) * house_size)).cuda()))  # 0按行，1按列
        sequence = torch.stack(seq_list, 0)  # month_len, batch_size, lstm_input_dim
        # 将数据放入LSTM训练
        out, hidden = self.lstm(sequence)  # out:(month_len, batch_size, hidden_size)
        out = out.view(shape, self.nfeat)
        x = self.linear_price(out)
        return x 