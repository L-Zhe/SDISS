import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
from    copy import deepcopy

class GCN(nn.Module):
    def __init__(self, inputs_size):
        super().__init__()
        self.w_graph = nn.Linear(inputs_size, inputs_size, bias=True)
        self.w_gate  = nn.Linear(inputs_size * 2, inputs_size, bias=True)
        self.info    = nn.Linear(inputs_size * 2, inputs_size, bias=True)
        self.w_a     = nn.Linear(inputs_size, 1, bias=True)
        self.w_h     = nn.Linear(2, 1, bias=True)
        self.d_k     = inputs_size ** 0.5
        self.gate    = nn.Linear(inputs_size, inputs_size, bias=True)
        nn.init.xavier_uniform_(self.w_graph.weight)
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_a.weight)
        nn.init.xavier_uniform_(self.w_h.weight)
        nn.init.xavier_uniform_(self.info.weight)

        nn.init.constant_(self.w_graph.bias, 0.)
        nn.init.constant_(self.w_gate.bias, 0.)
        nn.init.constant_(self.w_a.bias, 0.)
        nn.init.constant_(self.w_h.bias, 0.)
        nn.init.constant_(self.info.bias, 0.)
        
    def forward(self, stEmbed, edEmbed, edgeEmbed, mask):
        gate = torch.sigmoid(self.w_gate(torch.cat((edgeEmbed, edEmbed), dim=-1)))
        info = self.info(torch.cat((edgeEmbed, edEmbed), dim=-1)) * gate
        # score: [batch, seqLen, edgeLen]
        score = F.leaky_relu(self.w_h(torch.cat((self.w_a(stEmbed), self.w_a(info)), dim=-1)) \
                        .squeeze(-1)).masked_fill(mask, -np.inf)
        weight = F.softmax(score, dim=-1)
        a = (info * weight.unsqueeze(-1)).sum(dim=-2)

        outputs = self.w_graph(a / self.d_k)
        return F.relu(outputs)

class MultiHead(nn.Module):
    def __init__(self, inputs_size, num_head, FFN):
        super().__init__()
        self.w_head_st   = nn.Linear(inputs_size, inputs_size)
        self.w_head_ed   = nn.Linear(inputs_size, inputs_size)
        self.w_head_edge = nn.Linear(inputs_size, inputs_size)
        self.w_out       = nn.Linear(inputs_size, inputs_size, bias=True)
        self.FFN         = deepcopy(FFN)
        self.gcn         = GCN(int(inputs_size / num_head))
        self.layerNorm   = nn.LayerNorm(inputs_size)
        self.num_head    = num_head
        self.inputs_size = inputs_size

        nn.init.xavier_uniform_(self.w_head_st.weight)
        nn.init.xavier_uniform_(self.w_head_ed.weight)
        nn.init.xavier_uniform_(self.w_head_edge.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        nn.init.constant_(self.w_out.bias, 0.)
        
    
    def forward(self, nodeEmbed, edgeEmbed, edIndex, graph):
        # nodeEmbed: [batch, seqLen, hidden_size]
        # edEmbed: [batch, seqLen, edgeLen, hidden_size]
        # edgeEmbed: [batch, seqLen, edgeLen, hidden_size]
        # graph: [batch, seqLen, edgeLen, 2]
        mask = (graph[..., 0]<0)
        edgeLen = edgeEmbed.size(2)
        stEmbed = nodeEmbed.unsqueeze(-2).repeat(1, 1, edgeLen, 1)
        edEmbed = stEmbed.gather(1, edIndex)
        edEmbed.masked_fill_(mask.unsqueeze(-1).repeat(1, 1, 1, self.inputs_size), 0)

        batch = stEmbed.size(0)
        seqLen = stEmbed.size(1)
        edgeLen = stEmbed.size(2)

        stEmbed = self.w_head_st(stEmbed).view(batch, seqLen, edgeLen, self.num_head, -1).transpose_(2, 3)
        edEmbed = self.w_head_ed(edEmbed).view(batch, seqLen, edgeLen, self.num_head, -1).transpose_(2, 3)
        edgeEmbed = self.w_head_edge(edgeEmbed).view(batch, seqLen, edgeLen, self.num_head, -1).transpose_(2, 3)
        mask = mask.unsqueeze(-2).repeat(1, 1, self.num_head, 1)
        outputs = self.gcn(stEmbed, edEmbed, edgeEmbed, mask).view(batch, seqLen, -1)
        outputs = self.layerNorm(self.w_out(outputs) + nodeEmbed)
        return self.layerNorm(outputs + self.FFN(outputs))

class SEGCN(nn.Module):
    def __init__(self, edgeSize, embedding_size, num_layers, num_head, FFN, dropout_embed):
        super().__init__()
        self.forGCN         = nn.ModuleList([MultiHead(embedding_size, num_head, FFN) for i in range(num_layers)])
        self.backGCN        = nn.ModuleList([MultiHead(embedding_size, num_head, FFN) for i in range(num_layers)])
        self.Embedding      = nn.Embedding(edgeSize, embedding_size)
        self.dropout_embed  = nn.Dropout(dropout_embed)
        self.embedding_size = embedding_size

    def forward(self, forEmbed, graph, rgraph):
        # edgeEmbed: [batch, seqLen, edgeLen, hidden_size]
        mask = (graph[..., 1]<0)
        edgeEmbed = self.dropout_embed(self.Embedding(graph[..., 1].masked_fill(mask, 0)) \
                    .masked_fill(mask.unsqueeze(-1).repeat(1, 1, 1, self.embedding_size), 0))
        mask = (rgraph[..., 1] < 0)
        redgeEmbed = self.dropout_embed(self.Embedding(rgraph[..., 1].masked_fill(mask, 0)) \
                    .masked_fill(mask.unsqueeze(-1).repeat(1, 1, 1, self.embedding_size), 0))
        backEmbed = forEmbed.clone()


        edIndex = graph[...,0].masked_fill(graph[..., 1]<0, 0.).unsqueeze(-1).repeat(1, 1, 1, self.embedding_size)
        stIndex = rgraph[...,0].masked_fill(rgraph[..., 1]<0, 0.).unsqueeze(-1).repeat(1, 1, 1, self.embedding_size)

        for gcnLayer in self.forGCN:
            forEmbed = gcnLayer(forEmbed, edgeEmbed, edIndex, graph)

        for gcnLayer in self.backGCN:
            backEmbed = gcnLayer(backEmbed, redgeEmbed, stIndex, rgraph)
            
        return forEmbed + backEmbed
