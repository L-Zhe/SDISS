import  torch
from    torch import nn
from    torch.nn import functional as F
from    copy import deepcopy

class PositionWiseFeedForwardNetworks(nn.Module):
    def __init__(self, input_size, output_size, d_ff, dropout):
        super().__init__()
        self.W_1     = nn.Linear(input_size, d_ff, bias=True)
        self.W_2     = nn.Linear(d_ff, output_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W_1.weight)
        nn.init.xavier_uniform_(self.W_2.weight)
        nn.init.constant_(self.W_1.bias, 0.)
        nn.init.constant_(self.W_2.bias, 0.)
    
    def forward(self, inputs):
        outputs = F.relu(self.W_1(inputs))
        return self.W_2(self.dropout(outputs))

class EncoderCell(nn.Module):
    def __init__(self, d_model, Attention, FFNlayer, dropout):
        super().__init__()
        self.attn      = deepcopy(Attention)
        self.FFN       = deepcopy(FFNlayer)
        self.layerNorm = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, inputs, PadMask):
        inputs = self.layerNorm(inputs + self.attn(inputs, inputs, inputs, PadMask))
        return self.layerNorm(inputs + self.dropout(self.FFN(inputs)))



class DecoderCell(nn.Module):
    def __init__(self, d_model, Attention, FFNlayer, dropout):
        super().__init__()
        self.encodeAttn = deepcopy(Attention)
        self.gcnAttn    = deepcopy(Attention)
        self.maskedAttn = deepcopy(Attention)
        self.FFN        = deepcopy(FFNlayer)
        self.layerNorm  = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, inputs, encoder_outputs, gcn_outputs, PadMask, SeqMask):
        inputs = self.layerNorm(inputs + self.maskedAttn(inputs, inputs, inputs, SeqMask))
        inputs = self.layerNorm(inputs + self.encodeAttn(inputs, encoder_outputs, encoder_outputs, PadMask))
        inputs = self.layerNorm(inputs + self.gcnAttn(inputs, gcn_outputs, gcn_outputs, PadMask))
        return self.layerNorm(inputs + self.dropout(self.FFN(inputs)))

