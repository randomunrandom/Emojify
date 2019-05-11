import torch
from torch import nn
import torch.nn.functional as F

import sru

class SequentialCrossEntropy(nn.Module):
    def __init__(self, base_criterion=nn.CrossEntropyLoss(ignore_index=1)):
        super(SequentialCrossEntropy, self).__init__()
        self.criterion = base_criterion
        
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        batch_size = input.size(0)
        seq_len = input.size(1)
        
        return self.criterion(input=input.view(batch_size*seq_len, -1), target=target.view(batch_size*seq_len))

class SRUPos(nn.Module):
    def __init__(self, vocab_size, emojis_num, embedding_dim=256, hidden_dim=128, seq_len=30, embed_matrix=None, freeze=True):
        super(SRUPos, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.emojis_num = emojis_num
        
        if embed_matrix is None:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                padding_idx=1
            )
        else:
            self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze)
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.rnn = sru.SRU(embedding_dim, hidden_dim, layer_norm=True, use_relu=True, num_layers=2, dropout=0.5, rnn_dropout=0.5, bidirectional = True)
        self.output_layer = nn.Linear(in_features=2*hidden_dim, out_features=emojis_num)
        
        nn.init.kaiming_normal_(self.embedding.weight)
        nn.init.kaiming_normal_(self.output_layer.weight)
        
    def forward(self, x):
        emb = self.embedding(x)
        emb = self.norm(emb).permute(1, 0, 2)
        
        hidden, _ = self.rnn(emb)
        hidden = hidden.permute(1, 0, 2)
        
        output = self.output_layer(hidden)        
        return output
    
class SRUPosDouble(nn.Module):
    def __init__(self, vocab_size, emojis_num, embedding_dim=256, hidden_dim=128, hidden_dim2=64, seq_len=30, embed_matrix=None, freeze=True):
        super(SRUPosDouble, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.emojis_num = emojis_num
        
        if embed_matrix is None:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                padding_idx=1
            )
        else:
            self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze)
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.rnn = sru.SRU(embedding_dim, hidden_dim, layer_norm=True, use_relu=True, num_layers=2, dropout=0, rnn_dropout=0, bidirectional = True)
        self.rnn2 = sru.SRU(hidden_dim*2, hidden_dim2, layer_norm=True, use_relu=True, num_layers=2, dropout=0, rnn_dropout=0, bidirectional = True)
        self.output_layer = nn.Linear(in_features=2*hidden_dim2, out_features=emojis_num)
        
        nn.init.kaiming_normal_(self.embedding.weight)
        nn.init.kaiming_normal_(self.output_layer.weight)
        
    def forward(self, x):
        emb = self.embedding(x)
        emb = self.norm(emb).permute(1, 0, 2)
        emb = emb.permute(1, 0, 2)
        
        hidden, _ = self.rnn(emb)
        hidden, _ = self.rnn2(hidden)
        hidden = hidden.permute(1, 0, 2)
        
        output = self.output_layer(hidden)        
        return output