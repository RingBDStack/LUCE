# coding=utf-8
import numpy as np
import time
import os
import torch
import argparse
import torch.nn as nn
from model4 import *
from data4 import *
from utils import *


# ID，挂牌价，成交价，预测价，挂牌预测差，成交预测差
def price_str(val_predict, val_target, val_listprice):
    w_str = ''
    batch_size, label_size = val_predict.shape
    for k in range(batch_size):
        w_str += str(int(val_listprice[k,1]))+', '+str(val_listprice[k,0])+', '+str(val_target[k,0]) + \
                 ', ' + str(val_predict[k,0]) + ','  + str(abs(val_predict[k,0]-val_target[k,0])) + \
                 ', ' + str(abs(val_listprice[k,0]-val_target[k,0])) + '\n'
    return w_str


def save_parameters(file_path, file_name, args):
    with open(file_path+file_name, 'w') as f:
        f.write('data_path: '+str(args.data_path)+'\n')
        f.write('input_size: '+str(args.input_size)+'\n')
        f.write('layers: '+str(args.layers)+'\n')
        f.write('dropout: '+str(args.dropout)+'\n')
        f.write('epoch: '+str(args.epoch)+'\n')
        f.write('gc1_outdim: '+str(args.gc1_outdim)+'\n')
        f.write('gc2_outdim: '+str(args.gc2_outdim)+'\n')
        f.write('seq_len: '+str(args.seq_len)+'\n')
        f.write('meta_size: '+str(args.meta_size)+'\n')
        f.write('house_size: '+str(args.house_size)+'\n')
        f.write('lr: '+str(args.lr)+'\n')
        f.write('weight_decay: '+str(args.weight_decay)+'\n')
        f.write('set_data: '+str(args.set_data)+'\n')
        f.write('cuda: '+str(args.cuda)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../V5/Markham/")
    parser.add_argument("--gpu", type=str, default="0", choices=["0", "1"])
    parser.add_argument("--input_size", type=int, default=105)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=int, default=0.2)
    parser.add_argument("--epoch", type=int, default=1200)        # 训练周期
    parser.add_argument("--gc1_outdim", type=int, default=105)
    parser.add_argument("--gc2_outdim", type=int, default=105)
    parser.add_argument("--seq_len", type=int, default=37)       # 数据中的月数
    parser.add_argument("--house_size", type=int, default=350)   # 每个月的房屋数目
    parser.add_argument("--meta_size", type=int, default=4)      # 元路径数目
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument("--set_data", type=bool, default=False)
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()
    torch.cuda.manual_seed(30)
    # torch.manual_seed(42)
    # if args.cuda:
        # torch.cuda.manual_seed(42)

    # 数据读取
    if args.set_data:
        adj, features, labels, listprice, train_index, test_index = \
            load_data(path=args.data_path, month_len=args.seq_len, house_size=args.house_size)
        print('Data is generated.', flush=True)
    else:
        adj = np.load(args.data_path + 'adj.npy')
        features = np.load(args.data_path + 'features.npy')
        labels = np.load(args.data_path + 'labels.npy')
        listprice = np.load(args.data_path + 'listprice.npy')
        print('Data is loaded.', flush=True)
    print('adj: ' + str(adj.shape), flush=True)
    print('features: ' + str(features.shape), flush=True)
    print('labels: ' + str(labels.shape), flush=True)
    print('listprice: ' + str(listprice.shape), flush=True)
    print('***********************************************************', flush=True)

    # GPU设置：使用GPU或CPU
    if args.cuda:
        device = torch.device("cuda:" + args.gpu)
    else:
        device = torch.device('cpu')

    # 结果输出文件设置
    result_file_path = 'result_staticlstm/'
    model_file_path = 'model_saved_staticlstm/'
    other_file_path = 'result_staticlstm/others/'
    for output_path in [result_file_path, model_file_path, other_file_path]:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    save_parameters(result_file_path, 'parameters.txt', args)

    # 参数本地化
    train_epoch = args.epoch
    seq_len = args.seq_len
    whole_house_size = features.shape[0]
    feature_size = features.shape[1]
    input_dim = args.input_size
    house_size = args.house_size
    meta_size = args.meta_size
    gc1_outdim = args.gc1_outdim
    gc2_outdim = args.gc2_outdim

    train_index = range(10500)
    test_index = range(10500, 12950)

    adj = torch.tensor(adj).to(device)
    features = torch.tensor(features).to(device)
    labels = torch.tensor(labels).to(device)
    listprice = torch.tensor(listprice).to(device)
    train_index = torch.LongTensor(train_index).to(device)
    test_index = torch.LongTensor(test_index).to(device)
    print('adj: ' + str(adj.shape), flush=True)
    print('features: ' + str(features.shape), flush=True)
    print('labels: ' + str(labels.shape), flush=True)
    print('listprice: ' + str(listprice.shape), flush=True)
    print('train_index: ' + str(train_index.shape), flush=True)
    print('test_index: ' + str(test_index.shape), flush=True)

    model = GCNlstm_static(nfeat=input_dim, gc1_outdim=gc1_outdim, gc2_outdim=gc2_outdim,
                           dropout=args.dropout, meta_size=meta_size, house_size=house_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_criterion = nn.MSELoss()
    min_rmse = 10000
    min_train_loss = 10000
    w_str = ''
    # 暂定每个月的模型的训练周期相同
    for i in range(train_epoch):
        start_time = time.time()
        training_loss = []
        validation_losses = []
        # 开始训练
        model.train()
        optimizer.zero_grad()   # 梯度置零
        out_price = model(adj, features)  # LSTMs里的forward()函数，图卷积操作
        #print('out_price:' + str(out_price.shape))
        #print('labels[train_index]:' + str(labels[train_index].shape))
        loss = loss_criterion(out_price[train_index, 0], labels[train_index, 0])  # loss计算，pre与target
        loss.backward()     # 反向传播计算
        optimizer.step()    # 模型参数更新
        training_loss.append(loss.detach().cpu().numpy())
        avg_training_loss = sum(training_loss) / len(training_loss)
        print("Epoch:{}  Training loss:{}".format(i, avg_training_loss), flush=True)
        with open(result_file_path+'loss_error.txt', 'a+') as f:
            f.write("Epoch:{}  Training loss:{}\n".format(i, avg_training_loss))
        with open(other_file_path+'train_loss.txt', 'a+') as f:
            f.write("{}\n".format(avg_training_loss))

        # 对训练好的模型进行评估
        with torch.no_grad():
            model.eval()
            out_test_price = model(adj, features)
            val_predict = out_test_price[test_index].detach().cpu().numpy()
            val_target = labels[test_index].cpu().numpy()
            val_listprice = listprice[test_index].cpu().numpy()
            mse, mae, rmse = score(val_predict, val_target)
            y_pre_error = pre_error(val_predict, val_target)
            if rmse < min_rmse:
                min_rmse = rmse
                output = val_predict
                torch.save(model.state_dict(), model_file_path+'static.pkl')
                w_str = price_str(val_predict, val_target, val_listprice)
        end_time = time.time()
        cost_time = end_time-start_time
        # print(w_str)
        print("Test MSE: {} MAE:{} RMSE: {} pre_error:{} cost_time:{}".format(mse,mae,rmse,y_pre_error,cost_time), flush=True)
        with open(result_file_path+'loss_error.txt', 'a+') as f:
            f.write("Test MSE: {} MAE:{} RMSE: {} pre_error:{} cost_time:{}\n".format(mse,mae,rmse,y_pre_error,cost_time))
        with open(other_file_path + 'valid_RMSE.txt', 'a+') as f:
            f.write("{}\n".format(rmse))
        with open(other_file_path + 'pre_error.txt', 'a+') as f:
            f.write("{}\n".format(y_pre_error))
        with open(other_file_path+'price_list.csv', 'w') as f:
            f.write('index, list, target, pre, pre-target, list-target\n')
            f.write(w_str)




