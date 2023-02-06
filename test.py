from load_dataset import CreatDataset
from torch.utils.data import DataLoader
from model import *
import os
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

if not os.path.exists("result_img"):
    os.makedirs("result_img")

if not os.path.exists("training_model"):
    os.makedirs("training_model")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="which model", type=str, default="LS")
    parser.add_argument("--name", "-n", help="image name", type=str, default="Name")
    parser.add_argument("--epochs", "-e", help="total epoch", type=int, default=200)
    parser.add_argument("--step", "-st", help="print step", type=int, default=20)
    parser.add_argument("--source", "-s", help="csv path", type=str, default='./dataset/EURUSD_D_1.csv')
    parser.add_argument("--data_len", "-dl", help="data length", type=int, default=2000)
    args = parser.parse_args()
    return args

def test():
    args = parse_args()
    windows = 8
    file_path = args.source
    plt_name = str(args.model) + "_" + str(args.epochs) + "_" +  str(args.step)
    test_dataset = CreatDataset(file_path, windows, 'test',args.data_len, plt_name)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=2)

    model_dict = {
        "LS":LSTM,
        "BLS": Bi_LSTM,
        "LCA": LSTM_ChannelAttention,
        "LSA":LSTM_SpatialAttention,
        "LCBAM": LSTM_CBAM,
        "CL": CNN_LSTM,
        "CLCA": CNN_LSTM_ChannelAttention,
        "CLSA": CNN_LSTM_SpatialAttention,
        "CLCBAM": CNN_LSTM_CBAM,
    }

    model = model_dict[args.model](window=windows, dim=8, lstm_units=64, num_layers=2,hidden_size =200)
    criterion = nn.MSELoss()
    params = torch.load(f"training_model/{args.model}_best.pth")
    model.load_state_dict(params)

    # Initialization
    eval_loss = 0.0
    y_gt_test, y_pred = [], []
    with torch.no_grad():
        for data, label in test_loader:
            # Save test and predict data
            y_gt_test += label.numpy().squeeze(axis=1).tolist()
            out = model(data)
            loss = criterion(out, label)
            eval_loss += loss.item()
            y_pred += out.numpy().squeeze(axis=1).tolist()

    y_gt_test = np.array(y_gt_test)
    y_gt_test = y_gt_test[:, np.newaxis]
    y_pred = np.array(y_pred)
    y_pred = y_pred[:, np.newaxis]
    output_nor = pd.concat([pd.DataFrame(y_gt_test), pd.DataFrame(y_pred)], axis=1)

    # MAE
    print("Test MAE score: %.10f" % (mean_absolute_error(y_gt_test, y_pred)))
    # MSE
    print("Test MSE score: %.10f" % (mean_squared_error(y_gt_test, y_pred)))
    # RMSE
    print("Test RMSE score: %.10f" % (math.sqrt(mean_squared_error(y_gt_test, y_pred))))
    # R^2
    print("Test R^2 score: %.10f" % (r2_score(y_gt_test, y_pred)))

    # normalization(改命名)
    len_output = len(output_nor)
    df_csv_ori = pd.read_csv("./dataset/"+'8_column_data.csv')  #df_csv_ori是test*8列
    df_csv_data = df_csv_ori.loc[:, 'Open':'Close']
    #print(df_csv_data.shape)
    min_max_scaler = preprocessing.MinMaxScaler()
    #print(df_csv_data.shape)
    df_Normalization = min_max_scaler.fit_transform(df_csv_data)
    df_csv = pd.DataFrame(df_Normalization, columns=df_csv_data.columns)

    # inverse normalization
    df_close_win = df_csv['Close']
    df_close_win = df_close_win[0:len(df_close_win)-len(y_pred)]
    df_close_dl = pd.concat([df_close_win,pd.DataFrame(y_pred)])
    df_csv.pop('Close')  # 删除备注列
    df_csv.insert(df_csv.shape[1], 'Close', df_close_dl)  # 插入备注列
    df_inver = min_max_scaler.inverse_transform(df_csv) #
    df_inver = pd.DataFrame(df_inver,columns=['Open', 'High', 'Low', '10dRSI','MA5', 'MA20', 'Bias%', 'Close'])
    #print(df_inver)


    # plot inverse nor's data 改图名
    plt.figure(figsize=(12, 6))
    plt.xticks(fontsize=8)
    df_csv_date = pd.read_csv(file_path)  #读原本的四列数据取日期
    Date_Xaxis = df_csv_date['Date']
    Date_Xaxis = pd.to_datetime(Date_Xaxis.iloc[int(len(Date_Xaxis)) - len_output: int(len(Date_Xaxis))])
    plt.plot(Date_Xaxis, df_csv_ori['Close'][len(df_csv_ori)-len(y_pred):], Date_Xaxis, df_inver['Close'][len(df_csv_ori)-len(y_pred):])
    plt.legend(('real', 'predict'), loc='upper right', fontsize='15')
    plt.title("Test Data", fontsize='30')  # title
    plt.savefig("./result_img/"   + plt_name + "_" + "Test"+ "_inverse_"+str(int(args.data_len)*0.1)+".jpg", dpi=600)

    '''
    # plot 改图名
    plt.figure(figsize=(12, 6))
    plt.xticks(fontsize=8)
    plt.plot(Date_Xaxis, output_nor.iloc[:, 0], Date_Xaxis, output_nor.iloc[:, 1])
    plt.legend(('real', 'predict'), loc='upper right', fontsize='15')
    plt.title("Test Data", fontsize='30')  # title
    plt.savefig("./result_img/"  + plt_name +"_" + "Test" + "_normalization_"+ str(int(args.data_len)*0.1)+".jpg", dpi=1000)
    '''

    # 判断趋势预测是否符合,
    Acc = 0
    for i in range(0, len(output_nor.iloc[:, 0]) - 1):
        if ((float(output_nor.iloc[i + 1, 0]) - float(output_nor.iloc[i, 0])) > 0):
            if (float((output_nor.iloc[i + 1, 1]) - float(output_nor.iloc[i, 1])) > 0):
                Acc += 1
        else:
            if ((float(output_nor.iloc[i + 1, 1]) - float(output_nor.iloc[i, 1])) < 0):
                Acc += 1
    print('Test percent: {:.2%}'.format(Acc / len(output_nor.iloc[:, 0])), ",", Acc, "/", len(output_nor))

if __name__ in '__main__':
    test()
