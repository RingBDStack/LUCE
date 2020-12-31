import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np


'''
模型参数配置
'''
default_configure = {
    'data_path': '../V5/Markham/',
    'lstm_input_size': 105,
    'gc1_out_dim': 105,
    'layers': 1,
    'dropout': 0.2,  # 防止过拟合
    'epoch': 800,  # 训练周期
    'batch_size': 350,  # 批大小
    'seq_len': 37,  # 数据中的月数
    'house_size': 350,  # 每个月的房屋数目
    'meta_size': 4,  # 元路径数目
    'update_len': 4,  # 允许参数更新的月份长度
    'lr': 1e-3,             # 学习率
    'weight_decay': 5e-4,
    'set_data': False,  # 是否需要重新生成数据
}


def setup(args):
    args.update(default_configure)  # 更新参数
    args['device'] = 'cuda: 0' if torch.cuda.is_available() else 'cpu'   # GPU设置：使用GPU或CPU
    return args


def score(y_predict, y_target):
    y_predict = y_predict.reshape(1, -1)
    y_target = y_target.reshape(1, -1)
    mse = mean_squared_error(y_predict, y_target)
    mae = mean_absolute_error(y_predict, y_target)
    return mse, mae, np.sqrt(mse)


# 误差计算
def pre_error(y_predict, y_target):
    y_predict = y_predict.reshape(1, -1)
    y_target = y_target.reshape(1, -1)
    y_minor = y_predict-y_target
    y_minor = np.fabs(y_minor)
    y_error = np.true_divide(y_minor, y_target)
    y_avg_error = np.mean(y_error)
    pred_acc = r2_score(y_target.reshape(-1), y_predict.reshape(-1))
    return y_avg_error, pred_acc


# 从参数名称中提取当前月数
def get_layer(para_name):
    tmp = para_name.replace('glstm.','')
    nPos = tmp.index('.')
    num = int(tmp[:nPos])
    return num



