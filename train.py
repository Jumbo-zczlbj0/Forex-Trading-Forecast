from load_dataset import CreatDataset
from torch.utils.data import DataLoader
from model import *
import os
import math
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
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

def calculate_RMSE(gt_data, data):
    # Calculate RMSE
    y_gt_train = np.array(gt_data)
    y_gt_train = y_gt_train[:, np.newaxis]
    y_train = np.array(data)
    y_train = y_train[:, np.newaxis]
    output = pd.concat([pd.DataFrame(y_gt_train), pd.DataFrame(y_train)], axis=1)
    trainScore = math.sqrt(mean_squared_error(output.iloc[0:output.shape[0], 0], output.iloc[0:output.shape[0], 1]))
    return trainScore, output

def model_accuracy(output):
    # 判断趋势预测是否符合
    Acc=0
    for i in range (0,len(output.iloc[:, 0])-1):
        if ((float(output.iloc[i+1,0])-float(output.iloc[i,0])) >0) :
            if (float((output.iloc[i+1,1])-float(output.iloc[i,1])) >0) :
                Acc += 1
        else:
            if ((float(output.iloc[i+1,1])-float(output.iloc[i,1])) <0) :
                Acc += 1
    return Acc

def train():
    args = parse_args()
    windows = 8
    file_path = args.source

    # read train and val dataset
    plt_name = str(args.model) + "_" + str(args.epochs) + "_" +  str(args.step)
    train_data = CreatDataset(file_path, windows, 'train',args.data_len,plt_name)
    val_data = CreatDataset(file_path, windows, 'val',args.data_len,plt_name)
    # transfer data type
    train_loader = DataLoader(train_data, batch_size=200, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=200, shuffle=False, num_workers=2)

    model_dict = {
        "LS": LSTM,
        "BLS": Bi_LSTM,
        "LCA": LSTM_ChannelAttention,
        "LSA": LSTM_SpatialAttention,
        "LCBAM": LSTM_CBAM,
        "CL": CNN_LSTM,
        "SE": CNN_LSTM_SE,
        "ECA": CNN_LSTM_ECA,
        "CBAM": CNN_LSTM_CBAM,
        "HW": CNN_LSTM_HW
    }

    model = model_dict[args.model](window=windows, dim=8, lstm_units=64, num_layers=2, hidden_size = 150)
    print(model)
    print(f"training model is {args.model}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Initialization
    lose_list = []
    lose_list_train = []
    min_loss = float("inf")

    for epoch in tqdm(range(args.epochs)):
        running_loss = 0.0
        for step, (data, label) in enumerate(train_loader):
            out = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % args.step == 0:
                eval_loss_val, eval_loss_train = 0.0, 0.0
                with torch.no_grad():
                    y_pred_val, y_gt_val = [], []
                    y_pred_train, y_gt_train = [], []
                    mse_loss = 0.0
                    mse_loss_train = 0.0

                    for data, label in val_loader:
                        out = model(data)
                        loss = criterion(out, label)
                        mse_loss += loss.item()
                        # Save validation and predict data
                        y_gt_val += label.numpy().squeeze(axis=1).tolist()
                        out_val = model(data)
                        loss = criterion(out_val, label)
                        eval_loss_val += loss.item()
                        y_pred_val += out_val.numpy().squeeze(axis=1).tolist()
                    for data, label in train_loader:
                        out = model(data)
                        loss = criterion(out, label)
                        mse_loss_train += loss.item()
                        # Save training and predict data
                        y_gt_train += label.numpy().squeeze(axis=1).tolist()
                        out_train = model(data)
                        loss = criterion(out_train, label)
                        eval_loss_train += loss.item()
                        y_pred_train += out_train.numpy().squeeze(axis=1).tolist()

                    # Preserving the best training models
                    if mse_loss / len(val_loader) < min_loss:
                        torch.save(model.state_dict(), f"training_model/{args.model}_best.pth")
                        #print("save training model")
                        min_loss = mse_loss / len(val_loader)
                    #print(f"step:{step}, train loss:{mse_loss / len(val_loader)}")
                    lose_list.append(mse_loss / len(val_loader))
                    lose_list_train.append(mse_loss_train / len(train_loader))

    print("################## Training completed ##################")

    # MAE
    print("Val MAE score: %.10f" % (mean_absolute_error(y_gt_val, y_pred_val)))
    # MSE
    print("Val MSE score: %.10f" % (mean_squared_error(y_gt_val, y_pred_val)))
    # RMSE
    print("Val RMSE score: %.10f" % (math.sqrt(mean_squared_error(y_gt_val, y_pred_val))))
    # R^2
    print("Val R^2 score: %.10f" % (r2_score(y_gt_val, y_pred_val)))

    # Calculate val RMSE
    trainScore, output_train = calculate_RMSE(y_gt_train,y_pred_train)
    # Calculate train RMSE
    trainScore, output_val = calculate_RMSE(y_gt_val,y_pred_val)
    #以上两行可以删除，但是output_val需要更改169行等的输入值

    # train accuracy
    Acc = model_accuracy(output_train)
    print ('Train percent: {:.2%}'.format(Acc/len(output_train)),",",Acc,"/",len(output_train))
    # val accuracy
    Acc = model_accuracy(output_val)
    print ('Val percent: {:.2%}'.format(Acc/len(output_val)),",",Acc,"/",len(output_val))

    # plot loss（图名改）
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(lose_list_train)),lose_list_train,range(len(lose_list)),lose_list)
    plt.legend(('Train loss', 'Val loss'), loc='upper right', fontsize='15')
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title("Training&Val Loss", fontsize='30')  # title
    plt.savefig("./result_img/"  + plt_name + "_"  + "Train" +"_loss_"+str(int(args.data_len))+".jpg", dpi=600)

    # plot val(图名改)
    plt.figure(figsize=(12, 6))
    plt.xticks(fontsize=8)
    plt.plot(range(len(y_gt_val)), y_gt_val, range(len(y_gt_val)), y_pred_val)
    plt.legend(('real', 'predict'), loc='upper right', fontsize='15')
    plt.title("Val Data", fontsize='30')  # title
    plt.savefig("./result_img/"+ plt_name +"_" + "Train" + "_val_"+str(int(args.data_len*0.1))+".jpg", dpi=600)

if __name__ in '__main__':
    train()
