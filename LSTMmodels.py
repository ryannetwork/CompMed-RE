import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset
from torch.nn import functional as F
# Module for LSTM RNN


class LSTMsentence(nn.Module):
    def __init__(self, feature_dim, hidden_dim, batch_size, output_dim=1, bidirection=False, num_layers=1):
        super(LSTMsentence, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, self.num_layers)

        # max pooling layer
        self.maxpool = nn.MaxPool1d(1)
        self.out = nn.Linear(self.hidden_dim, 1)

        # hidden layer
        self.hidden = self.init_hidden()

        
    def init_hidden(self):
        # initialize the hidden state
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), 
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, X_train, Y_train, X_len):

        X = torch.nn.utils.rnn.pack_padded_sequence(X_train, X_len, batch_first=True)
        lstm_out, self.hidden = self.lstm(X, self.hidden)
        
        lstm_out = lstm_out.view(self.hidden_dim, -1)
        pool = self.maxpool(lstm_out)
        return F.log_softmax(pool, dim=1)


   

## data loader
class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, x_len):
        self.x = x_tensor
        self.y = y_tensor
        self.x_len = x_len
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.x_len[index])
    
    def __len__(self):
        return len(self.x)



        



    


