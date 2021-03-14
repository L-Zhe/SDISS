from    SemSS.Model import SemSS
import  Parameter

def createModel(sentence_vocab_size, graph_vocab_size):
    return SemSS(vocab_size        = sentence_vocab_size,
                 edge_size         = graph_vocab_size,
                 embedding_dim     = Parameter.embedding_dim,
                 d_ff              = Parameter.d_ff,
                 num_head          = Parameter.num_head,
                 num_layer_gcn     = Parameter.num_layer_gcn,
                 num_head_gcn      = Parameter.num_head_gcn,
                 num_layer_encoder = Parameter.num_layer_encoder,
                 num_layer_decoder = Parameter.num_layer_decoder,
                 dropout_embed     = Parameter.dropout_embed,
                 dropout_sublayer  = Parameter.dropout_sublayer,
                 dropout_ffn       = Parameter.dropout_ffn)
