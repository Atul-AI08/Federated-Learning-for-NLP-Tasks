import torch.nn as nn

class SentimentAnalyzer(nn.Module):
    """A LSTM based model 

    Args:
    + vocab_size
    + output_size
    + embedding_dim
    + hidden_dim
    + num_layers
    """
    def __init__(self,
                 vocab_size,
                 output_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers
                ):
        super(SentimentAnalyzer,self).__init__()
        self.output_size = output_size
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
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


def init_nets(embedding_dim, out_dim, vocab_size, n_parties, args, device="cpu"):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net = SentimentAnalyzer(vocab_size, out_dim, embedding_dim, args.hidden_dim, 1)
        # net = net.cuda()
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type