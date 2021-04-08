import  torch
from    torch import nn
from    torch.nn import functional as F
from    SDISS.EncoderDecoder import EncoderCell, DecoderCell, PositionWiseFeedForwardNetworks
from    SDISS.Embedding import Embedding
from    SDISS.Attention import MultiHeadAttention
from    SDISS.GCN import SEGCN
from    numpy import inf
from    copy import deepcopy

def pad_mask(inputs, PAD):
    return (inputs==PAD).unsqueeze(-1).unsqueeze(1)

def triu_mask(batch, length):
    mask = torch.ones(length, length).triu(1)
    return mask.unsqueeze(0).unsqueeze(1).bool()

class Model(nn.Module):

    def __init__(self, d_model, d_vocab, num_layer_encoder, num_layer_decoder, 
                 SEGCN, Embed, Attention, FFN, dropout_sublayer=0.):
        super().__init__()
        self.embedding = Embed
        self.SEGCN = SEGCN
        self.Encoder = nn.ModuleList([EncoderCell(d_model, Attention, FFN, dropout_sublayer) for _ in range(num_layer_encoder)])
        self.Decoder = nn.ModuleList([DecoderCell(d_model, Attention, FFN, dropout_sublayer) for _ in range(num_layer_decoder)])
        self.project = nn.Linear(d_model, d_vocab)
        self.w_eta = nn.Linear(d_model, 1, bias=True) 
        self.w_out = nn.Linear(2 * d_model, d_model, bias=True)
        self.d_model = d_model
        self.d_vocab = d_vocab

        nn.init.xavier_uniform_(self.project.weight)
        nn.init.xavier_uniform_(self.w_eta.weight)
        nn.init.xavier_uniform_(self.w_out.weight)

        nn.init.constant_(self.w_eta.bias, 0.)
        nn.init.constant_(self.w_out.bias, 0.)

    def copy(self, src_index, embed, encoder_outputs, mask):
        eta = torch.sigmoid(self.w_eta(embed))
        p_g = torch.matmul(embed, encoder_outputs.transpose(1, 2)).masked_fill_(mask, -inf)
        p_g = F.softmax(p_g, dim=-1) * (1 - eta)
        
        prob = F.softmax(self.project(embed), dim=-1) * eta 
        src_index = src_index.unsqueeze(1).repeat(1, embed.size(1), 1)
        return prob.scatter_add(2, src_index, p_g), p_g
    
    def teacher_forcing(self, src_index, target, encoder_outputs, gcn_outputs, PAD):
        PadMask = pad_mask(target, PAD).to(target.device)
        SeqMask = triu_mask(target.size(0), target.size(1)).to(target.device) + PadMask
        copyMask =  (src_index==PAD).unsqueeze(1)
        copyMask[:, :, 0] = True
        embed = self.embedding(target)

        for DecoderCell in self.Decoder:
            embed = DecoderCell(embed, encoder_outputs, gcn_outputs, PadMask, SeqMask)
        
        srcMask = (src_index==PAD).unsqueeze(1)
        copy_outputs = F.relu(self.w_out(torch.cat((encoder_outputs, gcn_outputs), dim=-1)))
        prob, attn_weight = self.copy(src_index, embed, copy_outputs, copyMask)

        return torch.log(prob+1e-15), attn_weight.sum(dim=-1)

    def greedy_search(self, src_index, encoder_outputs, gcn_outputs, maxLen, BOS, EOS, PAD):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        srcMask = (src_index==PAD).unsqueeze(1)
        sentence = torch.LongTensor([BOS]).repeat(batch_size).reshape(-1, 1).to(device)
        copy_outputs = F.relu(self.w_out(torch.cat((encoder_outputs, gcn_outputs), dim=-1)))
        copyMask = (src_index==PAD).unsqueeze(1)
        outputs = None
        EOS_index = torch.Tensor(batch_size).fill_(False).to(device)
        attn_weight = torch.LongTensor().to(device)
        for i in range(maxLen):

            embed = self.embedding(sentence)
            SeqMask = triu_mask(batch_size, i+1).to(device)
            for DecoderCell in self.Decoder:
                embed = DecoderCell(embed, encoder_outputs, gcn_outputs, None, SeqMask)
            prob, weight = self.copy(src_index, embed[:, -1:, :], copy_outputs, copyMask)
            attn_weight = torch.cat((attn_weight, weight.max(-1)[1]), dim=1)
            word = prob.max(dim=-1)[1].long()
            sentence = torch.cat((sentence, word), dim=1)
            
            EOS_index += (word==EOS).reshape(batch_size)
            if (EOS_index==False).sum() == 0:   break

        sentence = sentence.tolist()
        for i in range(batch_size):
            if EOS in sentence[i]:
                index = sentence[i].index(EOS)
                sentence[i] = sentence[i][:index]
            sentence[i] = sentence[i][1:]
        return sentence, attn_weight

    def beam_search(self, beam, src_index, encoder_outputs, gcn_outputs, maxLen, BOS, EOS, PAD):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        srcLen = encoder_outputs.size(1)
        
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, beam, 1, 1).view(-1, srcLen, self.d_model)
        gcn_outputs = gcn_outputs.unsqueeze(1).repeat(1, beam, 1, 1).view(-1, srcLen, self.d_model)
        copy_outputs = F.relu(self.w_out(torch.cat((encoder_outputs, gcn_outputs), dim=-1)))
        src_index = src_index.unsqueeze(1).repeat(1, beam, 1).view(-1, srcLen)
        copyMask =  (src_index==PAD).unsqueeze(1)
        copyMask[:, :, 0] = True
        sentence = torch.LongTensor(batch_size*beam, 1).fill_(BOS).to(device)
        EOS_index = torch.BoolTensor(batch_size,  beam).fill_(False).to(device)

        totalProb = torch.zeros(batch_size, beam).to(device)
        attn_weight = torch.LongTensor().to(device)
        for i in range(maxLen):
            embed = self.embedding(sentence)    # [Batch*beam, 1, hidden]
        
            SeqMask = triu_mask(batch_size  * beam, i+1).to(device)
            
            for DecoderCell in self.Decoder:
                embed = DecoderCell(embed, encoder_outputs, gcn_outputs, None, SeqMask)
            prob, weight = self.copy(src_index, embed[:, -1:, :], copy_outputs, copyMask) #[batch_size*beam, vocab]
            attn_weight = torch.cat((attn_weight, weight.max(-1)[1]), dim=1)
            prob = prob.squeeze(1)
            mask = EOS_index.view(batch_size*beam, 1)
            prob.masked_fill_(mask.repeat(1, self.d_vocab), 1)
            prob = torch.log(prob+1e-15) + totalProb.view(-1, 1)

            totalProb, index = prob.view(batch_size, -1).topk(beam, dim=-1, largest=True)
            # prob, index: [batch_size, beam]
            word = index % self.d_vocab
            index //= self.d_vocab
            # print(index.dtype)
            EOS_index = EOS_index.gather(dim=-1, index=index)
            EOS_index |= (word == EOS)
            if EOS_index.sum() == batch_size * beam:
                break
            
            sentence = sentence.view(batch_size, beam, -1).gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, i+1))
            sentence = torch.cat((sentence, word.unsqueeze(-1)), dim=-1).view(batch_size*beam, -1)
            attn_weight = attn_weight.view(batch_size, beam, -1).gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, i+1))
            attn_weight = attn_weight.view(batch_size*beam, -1)

        index = totalProb.max(1)[1]
        sentence = sentence.view(batch_size, beam, -1)
        attn_weight = attn_weight.view(batch_size, beam, -1)
        outputs = []
        attn = []
        for i in range(batch_size):
            sent = sentence[i, index[i], :].tolist()
            weight = attn_weight[i, index[i], :].tolist()
            if EOS in sent:
                sent = sent[:sent.index(EOS)]
            outputs.append(sent[1:])
            attn.append(weight)

        return outputs, attn

    def forward(self, inputs, targets, graph, rgraph, BOS, EOS, PAD):
        SrcPadMask = pad_mask(inputs, PAD).to(inputs.device)
        gcnEmbed = self.embedding(inputs)
        encodeEmbed = gcnEmbed.clone()
        for EncoderCell in self.Encoder:
            encodeEmbed = EncoderCell(encodeEmbed, SrcPadMask)
        gcnEmbed = self.SEGCN(gcnEmbed, graph, rgraph)
        outputs = self.teacher_forcing(inputs, targets, encodeEmbed, gcnEmbed, PAD)
        return outputs

    def predict(self, inputs, graph, rgraph, maxLen, PAD, BOS, EOS, search_method='greedy', beam_size=5):
        pos_length = inputs.size(1)
        SrcPadMask = pad_mask(inputs, PAD).to(inputs.device)
        gcnEmbed = self.embedding(inputs)
        encodeEmbed = gcnEmbed.clone()
        for EncoderCell in self.Encoder:
            encodeEmbed = EncoderCell(encodeEmbed, SrcPadMask)
        gcnEmbed = self.SEGCN(gcnEmbed, graph, rgraph)  
        if search_method == 'greedy':
            return self.greedy_search(inputs, encodeEmbed, gcnEmbed, maxLen, BOS, EOS, PAD)
        else:
            return self.beam_search(beam_size, inputs, encodeEmbed, gcnEmbed, maxLen, BOS, EOS, PAD)

def SemSS(vocab_size, edge_size, embedding_dim, d_ff, num_head, num_layer_gcn, num_head_gcn,
                num_layer_encoder, num_layer_decoder, dropout_embed=0.2, dropout_sublayer=0.1, dropout_ffn=0.):
    if embedding_dim % num_head != 0:
        raise ValueError("Parameter Error, require embedding_dim % num head == 0.")
    d_qk = d_v = int(embedding_dim / num_head)
    Embed = Embedding(vocab_size, embedding_dim)
    Attention = MultiHeadAttention(embedding_dim, d_qk, d_v, num_head)
    FFN = PositionWiseFeedForwardNetworks(embedding_dim, embedding_dim, d_ff, dropout_ffn)
    segcn = SEGCN(edge_size, embedding_dim, num_layer_gcn, num_head_gcn, FFN, dropout_embed)
    return Model(embedding_dim, vocab_size, num_layer_encoder, num_layer_decoder, 
                 segcn, Embed, Attention, FFN, dropout_sublayer)
