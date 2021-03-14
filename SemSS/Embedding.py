import  torch
from    torch import nn
import  math
import  numpy as np


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, dropout=0., maxLen=5000):
        super().__init__()
        self.embed   = nn.Embedding(num_embeddings, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('PE', self.PositionalEncoding(int(maxLen / 2), embedding_dim))
        self.embedding_dim = embedding_dim
        self.maxLen        = maxLen

        nn.init.xavier_uniform_(self.embed.weight)

    def PositionalEncoding(self, seq_length, embedding_dim):

        position = torch.arange(0., seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., embedding_dim, 2) * 
                             -(math.log(10000.0) / embedding_dim))
        tmp = position * div_term
        pe = torch.zeros(seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(tmp)
        pe[:, 1::2] = torch.cos(tmp)

        return pe.detach_()

    def forward(self, inputs):
        seq_length = inputs.size(1)
        if seq_length <= self.maxLen:
            outputs = self.embed(inputs) + self.PE[:seq_length]
        else:
            outputs = self.embed(inputs) \
                    + self.PositionalEncoding(seq_length, self.embedding_dim).to(inputs.device)
        return self.dropout(outputs)
