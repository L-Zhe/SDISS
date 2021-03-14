import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_qk, d_v, num_head):
        super().__init__()
        self.d_model  = d_model
        self.num_head = num_head
        self.d_qk     = d_qk
        self.d_v      = d_v
        self.W_Q      = nn.Linear(d_model, num_head * d_qk)
        self.W_K      = nn.Linear(d_model, num_head * d_qk)
        self.W_V      = nn.Linear(d_model, num_head * d_v)
        self.W_out    = nn.Linear(d_v * num_head, d_model)

        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

    def ScaledDotProductAttention(self, query, keys, values, mask):
        score = torch.matmul(query, keys.transpose(-1, -2)) / (self.d_model ** 0.5)
        if mask is not None:
            score.masked_fill_(mask, -np.inf)   
        weight = F.softmax(score, dim=-1)
        if mask is not None:
            weight = weight.masked_fill(mask, 0.)
        return torch.matmul(weight, values)

    def forward(self, Q, K, V, mask):
        batch_size = Q.size(0)
        query      = self.W_Q(Q).view(batch_size, Q.size(1), self.num_head, self.d_qk)
        keys       = self.W_K(K).view(batch_size, K.size(1), self.num_head, self.d_qk)
        values     = self.W_V(V).view(batch_size, V.size(1), self.num_head, self.d_v)
        query.transpose_(1, 2)
        keys.transpose_(1, 2)
        values.transpose_(1, 2)

        outputs = self.ScaledDotProductAttention(query, keys, values, mask)
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v*self.num_head)
        return self.W_out(outputs)