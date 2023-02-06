import os
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import pandas as pd
import numpy as np
import talib
from matplotlib import pyplot as plt

class CreatDataset(Dataset):
    def __init__(self, dataPath, window, name, data_len,plt_name):

        # calculate MA5,MA20,Bias%,10day RSI
        def moving_average(df, n1=5,n2=20):
            df['MA5'] = df['Close'].rolling(n1).mean()
            df['MA20'] = df['Close'].rolling(n2).mean()
            return df

        def Bias(df, L1=5):
            df['Bias%'] = 100 * (df['Close'] - df['Close'].rolling(L1).mean()) / df['Close'].rolling(L1).mean()
            return df

        # load csv file
        df_csv= pd.read_csv(dataPath)
        df_csv = df_csv.loc[:, 'Open':'Close']
        df_csv = df_csv.dropna(axis=0,how='any')  #Remove missing rows

        # calculate RSI,MA5,MA20,Bias
        df_csv['10dRSI'] = talib.RSI(df_csv.loc[:,"Close"],timeperiod=10)
        moving_average(df_csv, n1=5,n2=20)
        Bias(df_csv, 5)
        df_csv = df_csv.dropna(axis=0,how='any') #Remove missing rows

        #把Close列放到最后
        mid = df_csv['Close']  # 取备注列的值
        df_csv.pop('Close')  # 删除备注列
        df_csv.insert(df_csv.shape[1],'Close', mid)  # 插入备注列

        '''
        # choose different data points(1300,2600,4800)
        df_csv = df_csv.iloc[int(len(df_csv))-int(data_len) : int(len(df_csv)), :]
        print ('after_df_csv',len(df_csv),data_len,)

        # Dividing the data set
        if (name=='train'):
            df_csv = df_csv.iloc[0 : int(len(df_csv)*0.9), :]
        elif (name == 'test'):
            df_csv = df_csv.iloc[int(len(df_csv)*0.9) : int(len(df_csv)), :]
        else:
            print("Incorrect dataset name")
        '''

        # 把处理完的8列数据转存成csv文档
        if(os.path.exists("./dataset/"+'8_column_data.csv')):
            os.remove("./dataset/"+'8_column_data.csv')
        df_csv.to_csv("./dataset/"+'8_column_data.csv')
        print ('before_df_csv',len(df_csv))

        # draw image for original "MA5,MA20,Close" and "10Day RSI with y=30,70"
        if (name == 'test'):
            plt.figure(figsize=(12,6))
            plt.xlabel('Day')
            plt.xticks(fontsize=10)
            plt.plot(df_csv['Close'],'k-',df_csv['MA5'],'y--',df_csv['MA20'],'g-.')
            plt.plot(linewidth=1.0)
            plt.legend(('Close','MA5','MA20'), loc='upper right', fontsize='15')
            plt.title("Row Data & MA", fontsize='30')  # title
            plt.savefig("./result_img/"+"Row data"+"_"+ plt_name +"_" + str(int(0.1*data_len))+"_"+"test.jpg", dpi=600)

            plt.figure(figsize=(12,6))
            plt.xticks(fontsize=10)
            plt.plot(range(len(df_csv)),df_csv['10dRSI'],'c-')
            plt.ylim(0, 100)
            plt.hlines(y=30, xmin=0,xmax=len(df_csv))
            plt.hlines(y=70, xmin=0,xmax=len(df_csv))
            plt.plot(linewidth=1.0)
            plt.title("10 Days RSI", fontsize='30')  # title
            plt.savefig("./result_img/"+"10D_RSI"+"_"+ plt_name +"_"+ str(int(0.1*data_len))+"_"+"test.jpg", dpi=600)

        # Normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        df_Normalization = min_max_scaler.fit_transform(df_csv)
        df = pd.DataFrame(df_Normalization, columns=df_csv.columns)
        # Initialization
        df_forex = df
        # DataFrame to array
        data = df_forex.values
        dataset = []

        # Create the whole dataset
        for index in range(len(data) - window):  # 循环数据长度-sequence_length次
            dataset.append(data[index: (index + window + 1)])  # 第i行到i+sequence_length
        dataset = np.array(dataset)

        # choose different data points(2000,3000,4800)
        dataset = dataset[int(len(dataset))-int(data_len) : int(len(dataset)), :]

        # Dividing the data set
        if (name=='train'):
            dataset = dataset[0 : int(len(dataset)*0.8), :]
        elif (name == 'val'):
            dataset = dataset[int(len(dataset)*0.8) : int(len(dataset)*0.9), :]
        elif (name == 'test'):
            dataset = dataset[int(len(dataset)*0.9) : int(len(dataset)), :]
        else:
            print("Incorrect dataset name")

        # Create dataset and label
        X_data = dataset[:, :-1,:]
        Y_data = dataset[:,dataset.shape[1]-1,dataset.shape[2]-1]
        # save dataset and label
        self.data = X_data
        self.label = Y_data

    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self,idx): 
        return torch.from_numpy(self.data[idx]).to(torch.float32), torch.FloatTensor([self.label[idx]])
