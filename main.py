# main process
from WordEncoding import BioWordVec, RawData, TextEncoder
from LSTMmodels import LSTMsentence, CustomDataset
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

# parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#### 
all_instance = RawData()
word_model = BioWordVec()
print("Process into BioWordVec")
Big_tensor = []
for i in all_instance.data:
    txt = TextEncoder(i['txt'], word_model)
    new_array = txt.to_embeddings().reshape(txt.token_nums, -1)
    relation = i['r']['value']
    Big_tensor.append((new_array, relation))



####### task: convert relation to class num (done)
relations = list(set([i[1] for i in Big_tensor]))
rel_to_ix = {relations[i]:i for i in range(len(relations))}
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
X_len = torch.tensor(Big_sen_len)


###### Training / testing split
TRAIN_PART = 0.8
Big_tensor_train_index = random.sample(range(len(X_Big)), round(TRAIN_PART * len(X_Big)))
Big_tensor_test_index = [i for i in range(len(Big)) if i not in Big_tensor_train_index]
X_train = X_Big[Big_tensor_train_index, :, :]
X_test = X_Big[Big_tensor_test_index, :, :]



####### task: create costome Dataset instance and DataLoader
train_data = CustomDataset(x_train_tensor, y_train_tensor)
####### task: convert all data into PackedSequence Object 
### 





# settings for trainning
feature_dim = 200
hidden_dim = 100
batch_size = 128
net = LSTMsentence(feature_dim, hidden_dim, batch_size)


import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

# train
for epoch in range(EPOCHS):
    for data in Big_tensor_train:
        X, y = data
        net.zero_grad()
        print(type(X))
        output = net(X)
        loss = torch.exp(output)/torch.sum(torch.exp(output), dim=1).view(-1,1)
        loss.backward()
        optimizer.step()
    print(loss)







# def main():
    # Preprocess data
        # integrate raw into whole file

        # encode the data with BioWordVec

    # Training
        # initate LSTM for preceding, concept1, middle, concept 2, succeeding


        # train

    # visulize the results


