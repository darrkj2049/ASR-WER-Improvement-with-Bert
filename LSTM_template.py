import gensim
import torch
import torch.nn as nn

model = gensim.models.Word2Vec.load('./test_w2v.model')
weights = torch.FloatTensor(model.wv.vectors)


embedding = nn.Embedding.from_pretrained(weights)
embedding.requires_grad = False #to prevent training after data has been load

class LSTM(nn.Module):
    def init(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).init()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self,x):
        