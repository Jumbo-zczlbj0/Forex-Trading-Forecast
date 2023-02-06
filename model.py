import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, window=10, dim=8, lstm_units=64, num_layers=2, hidden_size = 64):
        super(LSTM, self).__init__()
        self.LSTM = nn.LSTM(lstm_units, hidden_size = hidden_size, batch_first=True, num_layers=num_layers)
        self.cls = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.reshape(x, ((x.shape[0], 1, 64)))  #把矩阵变成单列的数输入LSTM
        x, (_, _) = self.LSTM(x)
        x = self.cls(x[:, -1, :])  #官方代码
        return x

class Bi_LSTM(nn.Module):

    def __init__(self, window=10, dim=8, lstm_units=64, num_layers=2, hidden_size = 64):
        super(Bi_LSTM, self).__init__()
        self.LSTM = nn.LSTM(lstm_units, hidden_size = hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.cls = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = torch.reshape(x, ((x.shape[0], 1, 64)))  #把矩阵变成单列的数输入LSTM
        x, (_, _) = self.LSTM(x)
        x = self.cls(x[:, -1, :])  #官方代码
        return x


class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel=200, r=1):
        super(ChannelAttentionModul, self).__init__()

        self.MaxPool = nn.AdaptiveMaxPool2d(1)
        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_branch = self.MaxPool(x)
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        avg_branch = self.AvgPool(x)
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        h, w = weight.shape
        Mc = torch.reshape(weight, (h, w, 1, 1))
        x = Mc * x

        return x

class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        MaxPool = torch.max(x, dim=1).values
        AvgPool = torch.mean(x, dim=1)

        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        x_cat = torch.cat((MaxPool, AvgPool), dim=1)

        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        x = Ms * x

        return x

class LSTM_ChannelAttention(nn.Module):

    def __init__(self, window=10, dim=8, lstm_units=64, num_layers=2, hidden_size = 64):
        super(LSTM_ChannelAttention, self).__init__()
        self.LSTM = nn.LSTM(lstm_units, hidden_size = hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.Cam = ChannelAttentionModul(in_channel=200)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(hidden_size * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = torch.reshape(x, ((x.shape[0], 1, 64)))  #把矩阵变成单列的数输入LSTM
        x, (_, _) = self.LSTM(x)
        x = torch.unsqueeze(x,0)
        x = self.Cam(x)
        x = torch.squeeze(x,0)
        x = self.act2(x)
        x = self.cls(x[:, -1, :])  #官方代码
        x = self.act4(x)
        return x

class LSTM_SpatialAttention(nn.Module):

    def __init__(self, window=10, dim=8, lstm_units=64, num_layers=2, hidden_size = 64):
        super(LSTM_SpatialAttention, self).__init__()
        self.LSTM = nn.LSTM(lstm_units, hidden_size = hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.Sam = SpatialAttentionModul(in_channel=200)  # 空间注意力模块
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(hidden_size * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = torch.reshape(x, ((x.shape[0], 1, 64)))  #把矩阵变成单列的数输入LSTM
        x, (_, _) = self.LSTM(x)
        x = torch.unsqueeze(x,0)
        x = self.Sam(x)
        x = torch.squeeze(x,0)
        x = self.act2(x)
        x = self.cls(x[:, -1, :])  #官方代码
        x = self.act4(x)
        return x

class LSTM_CBAM(nn.Module):

    def __init__(self, window=10, dim=8, lstm_units=64, num_layers=2, hidden_size = 64):
        super(LSTM_CBAM, self).__init__()
        self.LSTM = nn.LSTM(lstm_units, hidden_size = hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.Cam = ChannelAttentionModul(in_channel=200)
        self.Sam = SpatialAttentionModul(in_channel=200)  # 空间注意力模块
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(hidden_size * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = torch.reshape(x, ((x.shape[0], 1, 64)))  #把矩阵变成单列的数输入LSTM
        x, (_, _) = self.LSTM(x)
        x = torch.unsqueeze(x,0)
        x = self.Cam(x)
        x = self.Sam(x)
        x = torch.squeeze(x,0)
        x = self.act2(x)
        x = self.cls(x[:, -1, :])  #官方代码
        x = self.act4(x)
        return x

class CNN_LSTM(nn.Module):

    def __init__(self, window=8, dim=8, lstm_units=64, num_layers=2, hidden_size = 64):
        super(CNN_LSTM, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, hidden_size = hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(hidden_size * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样,查一下是什么
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNN_LSTM_ECA(nn.Module):

    def __init__(self, window=8, dim=8, lstm_units=64, num_layers=2, hidden_size = 64):
        super(CNN_LSTM_ECA, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, hidden_size = hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.act2 = nn.Tanh()
        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.act3 = nn.Sigmoid()
        self.cls = nn.Linear(hidden_size * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        attn = self.attn(x)  # bs, 2*lstm_units  ECA_Attention是后面的3行。
        attn = self.act3(attn)
        x = x * attn
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNN_LSTM_SE(nn.Module):

    def __init__(self, window=8, dim=8, lstm_units=64, num_layers=2, hidden_size = 64):
        super(CNN_LSTM_SE, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, hidden_size = hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(hidden_size * 2, 1)
        self.act4 = nn.Tanh()
        self.se_fc = nn.Linear(window, window)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # se 前三行
        avg = x.mean(dim=1)  # bs, window
        se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
        x = torch.einsum("bnd,bd->bnd", x, se_attn)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNN_LSTM_CBAM(nn.Module):

    def __init__(self, window=10, dim=8, lstm_units=80, num_layers=2, hidden_size = 20):
        super(CNN_LSTM_CBAM, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, hidden_size = 20, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(hidden_size * 2, 1)
        self.act4 = nn.Tanh()
        self.se_fc = nn.Linear(window, window)
        self.hw_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # chanel
        avg = x.mean(dim=1)  # bs, window
        se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
        x = torch.einsum("bnd,bd->bnd", x, se_attn)
        # wh
        avg = x.mean(dim=2)  # bs, lstm_units
        hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
        x = torch.einsum("bnd,bn->bnd", x, hw_attn)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNN_LSTM_HW(nn.Module):

    def __init__(self, window=10, dim=8, lstm_units=80, num_layers=2, hidden_size = 20):
        super(CNN_LSTM_HW, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, hidden_size = 20, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(hidden_size * 2, 1)
        self.act4 = nn.Tanh()
        self.hw_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # wh
        avg = x.mean(dim=2)  # bs, lstm_units
        hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
        x = torch.einsum("bnd,bn->bnd", x, hw_attn)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x
