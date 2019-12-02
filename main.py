# main process
from WordEncoding import BioWordVec, RawData, TextEncoder
from LSTMmodels import LSTMsentence, CustomDataset
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F


# parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# settings for trainning
feature_dim = 200
hidden_dim = 100
batch_size = 10


#### 
all_instance = RawData()
word_model = BioWordVec()
print("Process into BioWordVec")
Big_tensor = []
for i in all_instance.data:
    txt = TextEncoder(i['txt'], word_model)
    new_array = txt.to_embeddings().reshape(txt.token_nums, -1)
    relation = i['r']['value']
    # y one hot array
    Big_tensor.append((new_array, relation))


####### task: convert relation to class num array (done)
relations = list(set([i[1] for i in Big_tensor]))
rel_to_ix = {relations[i]:i for i in range(len(relations))}
max_rel_len = len(relations)
def to_onehot(rel_ix, max_rel_len):
    outarray = np.zeros(max_rel_len)
    outarray[rel_ix] = 1
    return outarray


    
# Big = [(Big_tensor[i][0], to_onehot(rel_to_ix[Big_tensor[i][1]], max_rel_len)) for i in range(len(Big_tensor))] 
Big = [(Big_tensor[i][0], rel_to_ix[Big_tensor[i][1]]) for i in range(len(Big_tensor))] 




###### task: convert data into torch.tensor and get padded and stack into BIGGG tensor (Done)
Big = [(torch.tensor(i[0], requires_grad=True, dtype=torch.float).to(device), torch.tensor(i[1], requires_grad=True, dtype=torch.float).to(device)) for i in Big]

### get padded
Big_sen_len = [i[0].shape[0] for i in Big]
max_sen_len = max([i[0].shape[0] for i in Big])  
def pad_tensor(X, max_sen_len):
    m = torch.tensor(np.zeros((max_sen_len-X.shape[0], feature_dim)), requires_grad=True, dtype=torch.float) 
    return torch.cat([X, m], 0)
Big_p = [(pad_tensor(i[0], max_sen_len), i[1]) for i in Big]

### Big tensor real
X_Big = torch.stack([i[0] for i in Big_p])
Y_Big = torch.stack([i[1] for i in Big_p])
X_Len = torch.tensor(Big_sen_len)



###### Training / testing split
TRAIN_PART = 0.8
Big_tensor_train_index = random.sample(range(len(X_Big)), round(TRAIN_PART * len(X_Big)))
Big_tensor_test_index = [i for i in range(len(Big)) if i not in Big_tensor_train_index]
# train
X_train = X_Big[Big_tensor_train_index, :, :]
X_train_len = X_Len[Big_tensor_train_index,]
Y_train = Y_Big[Big_tensor_train_index,]
# train
X_test = X_Big[Big_tensor_test_index, :, :]
X_test_len = X_Len[Big_tensor_test_index,]
Y_test = Y_Big[Big_tensor_test_index, ]


####### task: create costom Dataset instance and DataLoader
train_data = CustomDataset(X_train, Y_train, X_train_len)
test_data = CustomDataset(X_test, Y_test, X_test_len)

### data loader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


####### task: Training!
net = LSTMsentence(feature_dim, hidden_dim, batch_size, len(relations), padding=max_sen_len)

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

# train
for epoch in range(EPOCHS):
    for data in train_loader:
        X_train, Y_train, X_len = data
        net.zero_grad()
        output = net(X_train, X_len)
        loss = torch.nn.CrossEntropyLoss()
        L = loss(output.view(batch_size,-1), Y_train.long())
        L.backward()
        optimizer.step()
    print(loss)


# dev
for data in train_loader:
    break
X_train, Y_train, X_len = data
net.zero_grad()
output = net(X_train, X_len)
loss = torch.nn.CrossEntropyLoss()
L = loss(output.view(batch_size,-1), Y_train.long())
L.backward()


# def main():
    # Preprocess data
        # integrate raw into whole file

        # encode the data with BioWordVec

    # Training
        # initate LSTM for preceding, concept1, middle, concept 2, succeeding


        # train

    # visulize the results


