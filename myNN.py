import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import config
param = config.Param()


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual  
        return self.relu(out)
    
    
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=param.residualNumLSTM):
        super(ResidualLSTM, self).__init__()
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size, 
                                                  hidden_size, 
                                                  batch_first=True,
                                                  num_layers=2,
                                                  dropout=param.dropoutLSTM) for _ in range(num_layers)])

    def forward(self, x):
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(x)
            out = out + x
            x = out
        return out

class LSTMResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=param.hiddenDim, num_residual_blocks=param.residualNumFC):
        super(LSTMResNet, self).__init__()
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers=1,batch_first=True)
        self.residual_lstm = ResidualLSTM(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_residual_blocks)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        x,(hn,cn) = self.lstm(x,(h0,c0))
        lstm_out = self.residual_lstm(x)
        out = lstm_out[:, -1, :] 
        for block in self.residual_blocks:
            out = block(out)
        out = self.fc(out)
        return out

def lstmresnet_test():
    batch_size = 1
    seq_length = 10
    input_dim = 10
    main_data = torch.randn(batch_size, seq_length, input_dim)
    hidden_dim = 256
    output_dim = 21
    num_layers = 6
    lstmModel = LSTMResNet(input_dim, output_dim)
    out= lstmModel(main_data,None,None)
    print(main_data.shape)
    print(out.shape)
    print(out)


if __name__ == "__main__":
    
    lstmresnet_test()



    pass





