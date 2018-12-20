import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    #def __init__(self, embedding_dim, hidden_dim, vocab_size):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers, drop_prob=0.2):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob)
        #self.lstm = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=drop_prob)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=1, padding=1) 
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.4)
        
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        batch_size = x.size(0)
        x = x.t()        
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        #print(embeds.shape)#[word_length, batch_size, embedding_dim]
        lstm_out, _ = self.lstm(embeds)
        #lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.view(batch_size, self.hidden_dim, -1)
        #print(lstm_out.shape) #[word_length, batch_size, lstm_output_size]
        
        conv_out = self.dropout(self.pool(F.relu(self.conv1(lstm_out))))
        conv_out = conv_out.view(-1, batch_size,  self.hidden_dim)
        out = self.dense(conv_out)
        out = out[lengths - 1, range(len(lengths))]
        
        return self.sig(out.squeeze())