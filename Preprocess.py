import  torch
from    torch import LongTensor
from    torch.utils.data import DataLoader, TensorDataset
import  Constants
import  random
import  numpy as np
import  pickle

def Vocab(filename_list, min_freq=0):

    corpus = []
    for filename in filename_list:
        with open(filename, 'r', encoding='UTF-8') as f:
            corpus.extend([word.lower() for seq in f.readlines() for word in seq.split()])
    vocab = set(corpus)
    wordCount = {}
    for word in corpus:
        if wordCount.get(word) is None:
            wordCount[word] = 1
        else:
            wordCount[word] += 1
        
    for word in list(vocab):
        if wordCount[word] <= min_freq:
            vocab.remove(word)
    print(len(vocab))
    vocab.add(Constants.PAD_WORD)
    vocab.add(Constants.BOS_WORD)
    vocab.add(Constants.EOS_WORD)
    vocab.add(Constants.UNK_WORD)
    vocab.add(Constants.SHORT_WORD)
    vocab.add(Constants.MIDDLE_WORD)
    vocab.add(Constants.LONG_WORD)
    vocab.add('.')
    
    word2index = {}; index2word={}
    word2index[Constants.PAD_WORD]    = Constants.PAD;    index2word[Constants.PAD]       = Constants.PAD_WORD
    word2index[Constants.BOS_WORD]    = Constants.BOS;    index2word[Constants.BOS]       = Constants.BOS_WORD
    word2index[Constants.EOS_WORD]    = Constants.EOS;    index2word[Constants.EOS]       = Constants.EOS_WORD
    word2index[Constants.UNK_WORD]    = Constants.UNK;    index2word[Constants.UNK]       = Constants.UNK_WORD
    word2index[Constants.SHORT_WORD]  = Constants.SHORT;  index2word[Constants.SHORT]     = Constants.SHORT_WORD
    word2index[Constants.MIDDLE_WORD] = Constants.MIDDLE; index2word[Constants.MIDDLE]    = Constants.MIDDLE_WORD
    word2index[Constants.LONG_WORD]   = Constants.LONG;   index2word[Constants.LONG]      = Constants.LONG_WORD

    for word in vocab:
        if word2index.get(word) is None:
            word2index[word] = len(word2index)
            index2word[len(index2word)] = word

    return word2index, index2word

def lang(filename, word2index, BOS=False, EOS=False, PAD=False, maxlen=None, include_lengths=False):

    def creatDataSet(filename, BOS, EOS, PAD, maxlen, include_lengths):
        with open(filename, 'r', encoding='UTF-8') as f:
            if maxlen is not None:
                corpus = [line.lower().split()[:maxlen] for line in f.readlines()]
            else:
                corpus = [line.lower().split() for line in f.readlines()]
            
            for i in range(len(corpus)):
                if corpus[i][-1] != '.':
                    if len(corpus[i]) < 2 or corpus[i][-2] != '.':
                        corpus[i].append('.')

            if BOS and EOS:
                corpus = [[Constants.BOS_WORD] + line + [Constants.EOS_WORD] for line in corpus]
            elif BOS:
                corpus = [[Constants.BOS_WORD] + line for line in corpus]
            elif EOS:
                corpus = [line + [Constants.EOS_WORD] for line in corpus]
            if PAD:
                lengths = [len(seq) for seq in corpus]
                maxlen = max(lengths)
                corpus = [line + [Constants.PAD_WORD] * (maxlen-len(line)) for line in corpus]
        if include_lengths:
            return corpus, lengths
        else:
            return corpus, None

    def prepare_sequence(corpus, word2idx):

        return [list(map(lambda w: word2idx[w] if word2idx.get(w) is not None
                    else word2idx[Constants.UNK_WORD], seq)) for seq in corpus]
    corpus, lengths = creatDataSet(filename, BOS, EOS, PAD, maxlen, include_lengths)
    
    return prepare_sequence(corpus, word2index)

            
def translate2word(sequence, index2word):
    return list(map(lambda word: index2word[word], sequence))

def graphVocab(file, min_freq=0.):

    wordCount = {}
    vocab = set()
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line[2:-3].split('), (')
            if len(line[0]) == 0: continue
            for elem in line:
                edge = elem.split(', ')[1].strip('\'')
                if edge == 'ROOT':  continue
                if edge in vocab:
                    wordCount[edge] += 1
                else:
                    vocab.add(edge)
                    wordCount[edge] = 1
    for key in wordCount.keys():
        if key in [Constants.PAD_WORD, Constants.UNK_WORD, 'self']:
            continue
        if wordCount[key] <= min_freq:
            vocab.remove(key)
    word2index = {}
    word2index[Constants.PAD_WORD] = Constants.PAD
    word2index[Constants.UNK_WORD] = Constants.UNK
    word2index['self'] = len(word2index)
    word2index['rself'] = len(word2index)
    for edge in vocab:
        word2index[edge] = len(word2index)
        word2index['r'+edge] = len(word2index)
    return word2index

    

def graph(file, gword2index, length):

    def creatSingleGraph(line, word2index, graphLen):
        line = line.strip()
        if len(line) == 2:  
            graph = [[[i, word2index['self']]] for i in range(graphLen)]
            rgraph = [[[i, word2index['rself']]] for i in range(graphLen)]
        else:
            line = line[2:-2].split('), (')
            length = max([max(int(elem.split(',')[0]), int(elem.split(',')[-1])) for elem in line]) + 1
            graph = [[[i, word2index['self']]] for i in range(graphLen)]
            rgraph = [[[i, word2index['rself']]] for i in range(graphLen)]
            for elem in line:
                _elem = elem.split(', ')
                edge = _elem[1].strip('\'')
                if edge == 'ROOT':  continue
                st = int(_elem[0])  # - 1
                ed = int(_elem[-1]) # - 1
                if word2index.get(edge) is not None:
                    graph[st].append([ed, word2index[edge]])
                    rgraph[ed].append([st, word2index['r' + edge]])
                else:
                    graph[st].append([ed, word2index[Constants.UNK_WORD]])
                    rgraph[ed].append([st, word2index[Constants.UNK_WORD]])

        return graph, rgraph
        
    graphs = []
    rgraphs = []
    with open(file, 'r') as f:
        for line in f.readlines():
            graph, rgraph = creatSingleGraph(line, gword2index, length+1)
            graphs.append(graph)
            rgraphs.append(rgraph)

    maxlen = max([len(line) for graph in graphs for line in graph])
    for i in range(len(graphs)):
        for j in range(len(graphs[i])):
            graphs[i][j].extend([[-1, -1] for _ in range(maxlen - len(graphs[i][j]))])
    maxlen = max([len(line) for graph in rgraphs for line in graph])
    for i in range(len(rgraphs)):
        for j in range(len(rgraphs[i])):
            rgraphs[i][j].extend([[-1, -1] for _ in range(maxlen - len(rgraphs[i][j]))])

    return graphs, rgraphs

def get_dataloader(source, graph, rgraph, targetInputs=None, targetOutputs=None, batch_size=64, shuffle=False):
    source = LongTensor(source)
    graph = LongTensor(graph)
    rgraph = LongTensor(rgraph)
    if targetInputs is not None and targetOutputs is not None:
        targetInputs = LongTensor(targetInputs)
        targetOutputs = LongTensor(targetOutputs)
        data = TensorDataset(source, graph, rgraph, targetInputs, targetOutputs)
    else:
        data = TensorDataset(source, graph, rgraph)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

def restore2Ori(inputs, dictPath):
    with open(dictPath, 'rb') as f:
        NER = pickle.load(f)
    NERtoken = {}

    for NERdict in NER:
        for key in NERdict.keys():
            if key != 'KEEP':
                NERtoken[key] = True
    
    def replaceWord(inputs, NERdict, NERtoken):
        for i in range(len(inputs)):
            if NERtoken.get(inputs[i].upper()) is None:
                continue
            elif NERdict.get(inputs[i].upper()) is None:
                inputs[i] = Constants.UNK_WORD
            else:
                inputs[i] = NERdict[inputs[i].upper()].lower()
        return inputs

    for i in range(len(inputs)):
        inputs[i] = replaceWord(inputs[i], NER[i], NERtoken)
    
    return inputs