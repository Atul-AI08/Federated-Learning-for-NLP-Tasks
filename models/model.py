import torch
import json
import torch.nn as nn
import numpy as np
from gensim.models import KeyedVectors
import os

GOOGLE_W2V = "./models/GoogleNews-vectors-negative300.bin"

class SentimentAnalyzer(nn.Module):
    """A LSTM based model 

    Args:
    + output_size
    + hidden_dim
    + num_layers
    """
    def __init__(self,
                 vocab_file,
                 vocab_size,
                 output_size,
                 hidden_dim,
                 num_layers
                ):
        super(SentimentAnalyzer,self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embed_vocab(vocab_file, vocab_size, 300))

        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        
        self.fc = nn.Sequential(nn.Linear(hidden_dim,32),
                                nn.ReLU(),
                                nn.Linear(32,4),
                                nn.Dropout(0.6),
                                nn.ReLU(),
                                nn.Linear(4,output_size))
        
    def forward(self,x):
        embeds = self.embedding(x)
        _,(hn,_) = self.lstm(embeds)
        return self.fc(hn[-1])
    

def embed_vocab(vocab_file, vocab_size, embedding_dim):
    with open(vocab_file,'r') as f:
        word2idx = json.load(f)

    word2vec = KeyedVectors.load_word2vec_format(GOOGLE_W2V, binary = True)
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    for word, i in word2idx.items():
        if word in word2vec:
            embedding_matrix[i] = word2vec[word]
    return torch.tensor(embedding_matrix, dtype = torch.float)


def init_nets(out_dim, vocab_file, vocab_size, n_parties, args, device="cpu"):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net = SentimentAnalyzer(vocab_file, vocab_size, out_dim, args.hidden_dim, 1)
        # net = net.cuda()
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type
