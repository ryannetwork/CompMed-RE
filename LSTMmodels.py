# Module for LSTM RNN


class LSTMsentence(nn.Module):
    def __init__(self, feature_dim, hidden_dim, embeddings, bidirection=False):
        self.__super__().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = embeddings

        # choice of bidirectional
        if bidirection:
            self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim)
    
    def getVec(self, sentence):
        np.array([self.embeddings.get_vector(word) for word in sentence])

    
    def forward(self, sentence):
        embeds = self.embeddings(sentence)




        



    





class LSTMsegment(nn.Module):
    
