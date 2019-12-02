# main process
from WordEncoding import BioWordVec, RawData, TextEncoder
from LSTMmodels import LSTMsentence
import random
# parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_dim = 
hidden_dim, 
batch_size
#### 
print("Process into BioWordVec")
Big_tensor = []
for i in all_instance.data:
    txt = TextEncoder(i['txt'], word_model)
    new_array = txt.to_embeddings().reshape(txt.token_nums, 1 , -1)
    relation = i['r']['value']
    Big_tensor.append((new_array, relation))


TRAIN_PART = 0.8
Big_tensor_train_index = random.sample(range(len(Big_tensor)), round(TRAIN_PART * len(Big_tensor)))
Big_tensor_test_index = [i for i in range(len(Big_tensor)) if i not in Big_tensor_train_index]
Big_tensor_train = [Big_tensor[i] for i in Big_tensor_train_index]
Big_tensor_test = [Big_tensor[i] for i in Big_tensor_test_index]



a = random.sample(range(len(Big_tensor)), round(TRAIN_PART * len(Big_tensor)))
Big_tensor_test_index = [i for i in range(len(Big_tensor)) if i not in Big_tensor_train_index]
Big_tensor_train = [Big_tensor[i] for i in Big_tensor_train_index]
Big_tensor_test = [Big_tensor[i] for i in Big_tensor_test_index]



feature_dim = all_instance[0].embedding_dim
hidden_dim = 100
batch_size = 128
net = LSTMsentence(feature_dim, hidden_dim, batch_size)

# train data
import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in Big_tensor_train:
        X, y = data
        net.zero_grad()
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


