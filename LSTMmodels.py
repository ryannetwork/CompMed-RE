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
        



    





class LSTMsegment(nn.Module):
    
