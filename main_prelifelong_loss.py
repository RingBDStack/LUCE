import argparse
from utils import *
from model4 import *
from data4 import *
import numpy as np
import time
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import math



# month * house => month * batch_num * batch_size
def make_index_batch(train_index, batch_size):
    month_len, house_size = train_index.shape
    train_index = torch.LongTensor(train_index)
    index_list = []
    for i in range(month_len):
        index_month_list = []
        for j in range(math.ceil(house_size / batch_size)):
            batch_start = j * batch_size
            batch_end = (j + 1) * batch_size
            # print('batch_start: ' + str(batch_start))
            # print('batch_end: ' + str(batch_end))
            if batch_end > house_size:
                batch_end = house_size
                batch_start = batch_end - batch_size
            index_month_list.append(train_index[i, batch_start:batch_end])
        index_month_list = torch.stack(index_month_list, 0)
        index_list.append(index_month_list)
    index_batch = torch.stack(index_list, 0).permute(1, 0, 2)
    return index_batch


def make_Y_from_index(labels, train_index):
    batch_num, month_len, batch_size = train_index.size()
    Y_train_batch = []
    for i in range(batch_num):
        Y_train_batch.append(labels[train_index[i]])
    Y_train_batch = torch.stack(Y_train_batch, 0)
    return Y_train_batch


# ID，挂牌价，成交价，预测价，挂牌预测差，成交预测差
def price_str(val_predict, val_target, val_listprice):
    w_str = ''
    #print('val_predict: ' + str(val_predict.shape))
    #print('val_target: ' + str(val_target.shape))
    #print('val_listprice: ' + str(val_listprice.shape))
    seq_len, batch_size, label_size = val_predict.shape
    for j in range(seq_len):
        for k in range(batch_size):
            w_str += str(int(val_listprice[j, k, 1]))+', '+str(val_listprice[j, k, 0])+', '+str(val_target[j, k, 0]) + \
                     ', ' + str(val_predict[j, k, 0]) + ',' + str(abs(val_predict[j, k, 0]-val_target[j, k, 0])) + \
                     ', ' + str(abs(val_listprice[j, k, 0]-val_target[j, k, 0])) + '\n'
    return w_str


def main(args):
    # 参数本地化
    train_epoch = args['epoch']
    seq_len = args['seq_len']
    gc1_out_dim = args['gc1_out_dim']
    lstm_input_dim = args['lstm_input_size']
    meta_size = args['meta_size']
    batch_size = args['batch_size']
    update_len = args['update_len']
    device = args['device']
    torch.cuda.manual_seed(42)
    # 数据读取
    if args['set_data']:
        adj, features, labels, listprice, train_index, test_index = \
            load_data(path=args['data_path'], month_len=seq_len, house_size=args['house_size'])
        print('Data is generated.', flush=True)
    else:
        adj = np.load(args['data_path'] + 'adj.npy')
        features = np.load(args['data_path'] + 'features.npy')
        labels = np.load(args['data_path'] + 'labels.npy')
        listprice = np.load(args['data_path'] + 'listprice.npy')
        train_index = np.load(args['data_path'] + 'train_index.npy')
        test_index = np.load(args['data_path'] + 'test_index.npy')
        print('Data is loaded.', flush=True)
    whole_house_size = features.shape[0]
    feature_size = features.shape[1]
    hidden_dim = feature_size  # 令hidden_dim与embedding的维度一致
    all_month = train_index.shape[0]+1
    house_size = int(whole_house_size/all_month)
    # 去除预训练已包含的部分
    train_index = train_index[-6:]
    test_index = test_index[-6:]
    print('adj: ' + str(adj.shape), flush=True)
    print('features: ' + str(features.shape), flush=True)
    print('labels: ' + str(labels.shape), flush=True)
    print('listprice: ' + str(listprice.shape), flush=True)
    print('train_index: ' + str(train_index.shape), flush=True)
    print('test_index: ' + str(test_index.shape), flush=True)
    print('***********************************************************', flush=True)
    # 结果输出文件设置
    result_file_path = 'result_prelifelong/'
    model_file_path = 'model_saved_prelifelong/'
    other_file_path = 'result_prelifelong/others/'
    # 如果目录不存在，则创建
    for output_path in [result_file_path, model_file_path, other_file_path]:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    # 数据批处理
    train_index_batch = make_index_batch(train_index, batch_size)
    print("train_index_batch: " + str(train_index_batch.shape), flush=True)
    test_index_batch = make_index_batch(test_index, house_size)
    print("test_index_batch: " + str(test_index_batch.shape), flush=True)
    # tensor化
    train_index_batch = train_index_batch.to(device)
    test_index_batch = test_index_batch.to(device)
    adj = torch.tensor(adj).to(device)
    features = torch.tensor(features).to(device)
    labels = torch.tensor(labels).to(device)
    listprice = torch.tensor(listprice).to(device)

    # 模型训练
    for cur_month in range(1, 7):
        # 一个月对应一个模型model，均在本月的模型内进行参数更新；cur_month代表当前参加训练的最后一月
        # r_gcnLSTMs每次都从送入数据的第一个月开始训练，逐步扩张模型至cur_month长度
        # 根据update_len，当cur_month超过update_len时，每次只更新[cur_month-update_len: cur_month]月的参数
        if cur_month <= update_len:
            model_lstm_len = cur_month
            train_index_p = train_index_batch[:, 0: cur_month, :]
            test_index_p = test_index_batch[:, 0: cur_month, :]
        else:
            model_lstm_len = update_len
            train_index_p = train_index_batch[:, cur_month - model_lstm_len: cur_month, :]
            test_index_p = test_index_batch[:, cur_month - model_lstm_len: cur_month, :]

        #print('train_index_p: ' + str(train_index_p.shape))
        #print('test_index_p: ' + str(test_index_p.shape))
        Y_train_batch = make_Y_from_index(labels, train_index_p).to(device)
        Y_test_batch = make_Y_from_index(labels, test_index_p).to(device)
        lp_batch = make_Y_from_index(listprice, test_index_p).to(device)
        batch_num = train_index_batch.shape[0]
        #print('Y_train_batch: ' + str(Y_train_batch.shape))
        #print('Y_test_batch: ' + str(Y_test_batch.shape))
        #print('lp_batch: ' + str(lp_batch.shape))

        # 给定参数，使得经过GCN和lstm的数据维度并不发生变化
        '''
        model = r_gcn_1LSTMs(gcn_input_dim=feature_size, lstm_input_dim=feature_size, hidden_dim=hidden_dim,
                           label_out_dim=1, Nodes=whole_house_size, meta_size=meta_size, all_month=all_month,
                           month_len=model_lstm_len, layers=args['layers'], dropout=args['dropout']
                           ).to(device)
        '''
        model = r_gcn2lv_1LSTMs(gcn_input_dim=feature_size, gc1_out_dim=gc1_out_dim, lstm_input_dim=feature_size,
                                hidden_dim=hidden_dim, label_out_dim=1,  meta_size=meta_size, all_month=all_month,
                                month_len=model_lstm_len, layers=args['layers'], dropout=args['dropout']).to(device)
        # 预训练模型参数载入
        if cur_month == 1:
            static_model = torch.load('model_saved_staticlstm/static.pkl')
            model_dict = model.state_dict()
            # 已有参数全部继承，包括LSTM和各月GCN
            state_dict = {'glstm.0.'+str(k): v for k, v in static_model.items() if 'glstm.0.'+str(k) in model_dict.keys()}
            print(state_dict.keys(), flush=True)
            model_dict.update(state_dict)
            state_dict = {k: v for k, v in static_model.items() if k in model_dict.keys()}
            print(state_dict.keys(), flush=True)
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif 1 < cur_month <= update_len:     # 当前月在更新范围内
            # 参数继承
            old_model = torch.load(model_file_path + 'month' + str(cur_month - 1) + '.pkl')
            model_dict = model.state_dict()
            # 已有参数全部继承，包括LSTM和各月GCN
            state_dict = {k: v for k, v in old_model.items() if k in model_dict.keys()}
            print(state_dict.keys(), flush=True)
            model_dict.update(state_dict)

            # 该月GCN模型参数沿用其前一个月的
            new_dict = {k.replace('glstm.' + str(int(cur_month - 2)), 'glstm.' + str(int(cur_month - 1))): v for k, v in
                        old_model.items() if 'glstm.' + str(int(cur_month - 2)) in k}
            model_dict.update(new_dict)

            model.load_state_dict(model_dict)
        elif cur_month > update_len:  # 当前月超出更新范围
            # 参数继承
            old_model = torch.load(model_file_path + 'month' + str(cur_month - 1) + '.pkl')
            model_dict = model.state_dict()
            # 已有参数全部继承，包括LSTM和各月GCN
            state_dict = {k: v for k, v in old_model.items() if k in model_dict.keys()}
            print(state_dict.keys(), flush=True)
            model_dict.update(state_dict)
            # 各月GCN错位继承，即old_model中的glstm.1应是model中的glstm.0
            gcn_dict = {k.replace('glstm.' + str(get_layer(k)), 'glstm.' + str(get_layer(k) - 1)): v
                          for k, v in old_model.items() if 'glstm.' in k and get_layer(k) > 0}
            print(gcn_dict.keys(), flush=True)
            model_dict.update(gcn_dict)
            model.load_state_dict(model_dict)

        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        loss_criterion = nn.MSELoss()
        min_rmse = 10000
        w_str = ''
        with open(other_file_path + 'month' + str(cur_month) + '_price_list.csv', 'a+') as f:
            f.write('index, list, target, pre, pre-target, list-target\n')
        
        # 暂定每个月的模型的训练周期相同
        for i in range(train_epoch*model_lstm_len):
            for b in range(batch_num):
                start_time = time.time()
                training_loss = []
                validation_losses = []
                model.train()
                optimizer.zero_grad()  # 梯度置零
                new_embedding, out_price = model(adj, features, train_index_p[b])
                # new_embedding = Variable(new_embedding.data, requires_grad=True)
                # features = new_embedding.to(device)  # features相当于全局变量，每次都继承
                # 送入的是多个月的index
                #print('out_price: ' + str(out_price.shape))
                #print('Y_train_batch[b]: ' + str(Y_train_batch[b][cur_month-1:cur_month].shape))
                #with torch.no_grad():
                  # weights = np.tanh(np.arange(1,Y_train_batch[b].shape[0]+1) * (np.e / Y_train_batch[b].shape[0]))
                  #weights = np.arange(1,Y_train_batch[b].shape[0]+1)* (1 / Y_train_batch[b].shape[0])
                  #weights = torch.tensor(weights, dtype=torch.float32, device=device)
                # print(weights)
                #loss = (out_price-Y_train_batch[b])**2
                # T = T.view(model_lstm_len,batch_size)
                for k in range(weights.shape[0]):
                  loss[k] = loss[k]*weights[k]
                # print('loss shape', loss.shape)
                # exit()
                # loss = ((out_price-Y_train_batch[b])**2).permute(1,0,2).squeeze()*weights.permute(1,0)  # loss计算，pre与target
                # T = (out_price-Y_train_batch[b])**2
                # print('T:')
                # print(T.shape)
                # print(loss.shape)
                loss = loss.mean()
                loss.backward()  # 反向传播计算
                optimizer.step()  # 模型参数更新
                training_loss.append(loss.detach().cpu().numpy())
                avg_training_loss = sum(training_loss) / len(training_loss)
                print("Month:{}  Epoch:{}  Training loss:{}".format(cur_month, i, avg_training_loss), flush=True)
                with open(result_file_path + 'month' + str(cur_month) + '_loss_error.txt', 'a+') as f:
                    f.write("Month:{}  Epoch:{}  Training loss:{}\n".format(cur_month, i, avg_training_loss))
                with open(other_file_path + 'train_loss.txt', 'a+') as f:
                    f.write("{}\n".format(avg_training_loss))

                # 对训练好的模型在测试集上进行评估
                with torch.no_grad():
                    model.eval()
                    _, out_test_price = model(adj, features, test_index_p[0])

                    val_target = Y_test_batch[0].cpu().numpy()
                    val_listprice = lp_batch[0].cpu().numpy()
                    val_predict = out_test_price.detach().cpu().numpy()
                    '''
                    print('test_index_p[0][-1:]: '+str(test_index_p[0][-1:].shape))
                    print('val_predict: '+str(val_predict.shape))
                    print('val_listprice: '+str(val_listprice.shape))
                    print('val_target: '+str(val_target.shape))
                    '''
                    # print('val_target: '+str(val_target.shape))
                    mse, mae, rmse = score(val_predict, val_target)
                    y_pre_error = pre_error(val_predict, val_target)
                    if rmse < min_rmse:
                        min_rmse = rmse
                        output = val_predict
                        torch.save(model.state_dict(), model_file_path + 'month' + str(cur_month) + '.pkl')
                        w_str = price_str(val_predict, val_target, val_listprice)
                        # features = new_embedding.to(device)
                end_time = time.time()
                cost_time = end_time - start_time
                print("Test MSE: {} MAE:{} RMSE: {} pre_error:{} cost_time:{}".format(mse, mae, rmse, y_pre_error, cost_time), flush=True)
                with open(result_file_path + 'month' + str(cur_month) + '_loss_error.txt', 'a+') as f:
                    f.write("Test MSE: {} MAE:{} RMSE: {} pre_error:{} cost_time:{}\n".format(mse, mae, rmse, y_pre_error, cost_time))
                with open(other_file_path + 'valid_RMSE.txt', 'a+') as f:
                    f.write("{}\n".format(rmse))
                with open(other_file_path + 'pre_error.txt', 'a+') as f:
                    f.write("{}\n".format(y_pre_error))

        with open(other_file_path + 'month' + str(cur_month) + '_price_list.csv', 'a+') as f:
            f.write(w_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GLSTM')
    args = parser.parse_args().__dict__
    args = setup(args)
    print('参数配置:\n{}'.format(args), flush=True)
    if not os.path.isdir('result_prelifelong/'):
        os.makedirs('result_prelifelong/')
    with open('result_prelifelong/parameters.txt', 'w') as f:
        f.write('Parameters:\n{}'.format(args))
    main(args)