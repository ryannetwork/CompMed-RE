# Basic attempt to replicate the RNN from "https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79"


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class SingleRNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(SingleRNN, self).__init__()

        self.Wx = torch.randn(n_inputs, n_neurons) # 4 x 1
        self.Wy = torch.randn(n_neurons, n_neurons) # 1 x 1

        self.b = torch.zeros(n_inputs, 1) # 4 x 1
    
    def forward(self, X0, X1):
        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b)

        self.Y1 = torch.tanh(torch.mm(self.Y0, self.Wx) + torch.mm(X1, self.Wx)
                                + torch.mm(X1, self.Wx) + self.b)
        return self.Y0, self.Y1


N_INPUT = 4
N_NEURONS = 1
X0_batch = torch.tensor([[0,1,2,0],
                         [3,4,5,0],
                         [6,7,8,0],
                         [9,0,1,0]], dtype=torch.float)
X1_batch = torch.tensor([[9,8,7,0],
                         [0,0,0,0],
                         [6,5,4,0],
                         [3,2,1,0]], dtype=torch.float)

model = SingleRNN(N_INPUT, N_NEURONS)
Y0_val, Y1_val = model.forward(X0_batch, X1_batch)
print("Y0_val: ", Y0_val)
print("Y1_val: ", Y1_val)