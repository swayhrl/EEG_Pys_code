from glob import glob
import scipy.io
import torch.nn as nn
import torch
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

class Net1(nn.Module):
    def __init__(self, in_channels,out_channels,hidden_size,output_size):
        super(Net1, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=30, seq_len=24, input_size=7) ---> permute(0, 2, 1)
        # (30, 7, 24)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        # (batch_size=30, out_channels=32, seq_len-4=20) ---> permute(0, 2, 1)
        # (30, 20, 32)
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.flatten(start_dim=2)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.flatten(start_dim=1) #5*192
        return x

class Net2(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Net2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=30, seq_len=24, input_size=7) ---> permute(0, 2, 1)
        # (30, 7, 24)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        self.fc = nn.Linear(68, 3)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.fc(x)
        x = x.flatten(start_dim=1)
        return x #5*192

class Net3(nn.Module):
    def __init__(self, in_channels,out_channels,output_size):
        super(Net3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        self.linear1 = torch.nn.Linear(324, 64)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, output_size)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        return x #5*192

class All_Net(nn.Module):
    def __init__(self):
        super(All_Net, self).__init__()
        self.net1 = Net1(320,128,128,64)
        self.net2 = Net2(5, 64)
        self.net3 = Net3(2, 64, 3)
        self.linear1 = torch.nn.Linear(576, 128)
        self.linear2 = torch.nn.Linear(128, 10)

    def forward(self, x1,x2,x3):
        output1 = self.net1(x1)
        output2 = self.net2(x2)
        output3 = self.net3(x3)
        output = torch.hstack([output1, output2, output3])
        output = self.linear1(output)
        output = self.linear2(output)
        return output  # batch_size*10

