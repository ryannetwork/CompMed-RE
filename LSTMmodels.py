


# Module for LSTM RNN


class LSTMsentence(nn.Module):
    def __init__(self, feature_dim, hidden_dim, embeddings, batch_size, output_dim=1, bidirection=False):
        super(LSTMsentence, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = embeddings
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, self.num_layers)

        # max pooling layer
        self.maxpool = nn.MaxPool1d(hidden_dim)

        # hidden layer
        self.hidden = self.init_hidden()

        
    def init_hidden(self):
        # initialize the hidden state
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), 
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, tokens):
        lstm_out, _ = self.lstm(tokens)
        lstm_out = lstm_out.view(len(tokens), -1)
        pool = max(lstm_out)
        return F.log_softmax(pool, dim=1)


    
    def forward(self, sentence):
        embeds = self.embeddings(sentence)



## data loader



        



    





class LSTMsegment(nn.Module):
    
