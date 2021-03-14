import  torch
import  Constants
from    Preprocess import translate2word, restore2Ori
from    Parameter import device, maxLen

def generator(model, testData, SrcOriPath, index2word, DictPath, search_method, beam_size):
       model.eval()
       translate = lambda outputs: [translate2word(seq, index2word) for seq in outputs]
       outputs = []
       attn_weight = []
       with torch.no_grad():
           for data in testData:
               if len(data) == 3:
                   source, graph, rgraph = data
               else:
                   source, graph, rgraph, _, _ = data
               source = source.to(device)
               lentok = torch.LongTensor(source.size(0), 1).fill_(Constants.MIDDLE).to(device)
               source = torch.cat((lentok, source), dim=-1)
               graph  = graph.to(device)
               rgraph = rgraph.to(device)
               sentence, attn = model.predict(source, graph, rgraph, maxLen,
                                              Constants.PAD, Constants.BOS, Constants.EOS,
                                              search_method=search_method, beam_size=beam_size)
               outputs.extend(translate(sentence))
               attn_weight.extend(attn)
       outputs = restore2Ori(outputs, DictPath)
       return replaceUNK(SrcOriPath, outputs, attn_weight)


def replaceUNK(srcPath, sentence, attn):
    with open(srcPath, 'r') as f:
        source = [[word.lower() for word in line.split()] for line in f.readlines()]
    for i in range(len(sentence)):
        for j in range(len(sentence[i])):
            if sentence[i][j] == Constants.UNK_WORD:
                sentence[i][j] = source[i][attn[i][j] - 1]
    return sentence


if __name__ == '__main__':
    pass