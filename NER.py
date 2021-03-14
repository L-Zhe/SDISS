from    stanfordcorenlp import StanfordCoreNLP
import  os
import  pickle
from    tqdm import tqdm

path = '/home/linzhe/tool/stanford-corenlp/stanford-corenlp-full-2018-10-05'
nlp = StanfordCoreNLP(path, lang='en')

def is_number(n):
    for i in range(len(n)):
        if n[i].isdigit():
            return True
    return False

def word2NER(ner):
    word2ners = [];  ner2words = []
    for elem in ner:
        word2ner = dict(elem)
        nerCount = {}
        for key in word2ner:
            word_ner = word2ner[key]
            if word_ner != 'O':
                if nerCount.get(word_ner) is None:
                    nerCount[word_ner] = 1
                    word2ner[key] = word_ner + '@0'
                else:
                    word2ner[key] = word_ner +'@' + str(nerCount[word_ner])
                    nerCount[word_ner] += 1
            else:
                word2ner[key] = 'KEEP'
        word2ners.append(word2ner)
        ner2words.append(dict(zip(word2ner.values(), word2ner.keys())))
    return word2ners, ner2words

def replaceNER(sent, word2ner):
    for i in range(len(sent)):
        if word2ner.get(sent[i]) is not None and \
            word2ner[sent[i]] != 'KEEP' :
            sent[i] = word2ner[sent[i]]
    return sent

def saveCorpus(corpus, file):
    with open(file, 'a') as f:
        for i in range(len(corpus)):
            f.write(' '.join(corpus[i]))
            if i != len(corpus) - 1:
                f.write('\n')

def saveNER(ner, file):
    with open(file, 'a') as f:
        for i in tqdm(range(len(ner))):
            f.write(str(ner[i]))
            if i != len(ner) - 1:
                f.write('\n')

def NERPaser(src, tgt):
    if os.path.exists(src + '.ner'):
        os.remove(src + '.ner')
    if os.path.exists(src + '.ner_parse'):
        os.remove(src + '.ner_parse')
    if os.path.exists(tgt + '.ner'):
        os.remove(src + '.ner')
    if os.path.exists(tgt + '.ner_parse'):
        os.remove(tgt + '.ner_parse')

    with open(src, 'r') as f:
        src_corpus = [line for line in f.readlines()]
    with open(tgt, 'r') as f:
        tgt_corpus = [line for line in f.readlines()]

    ner=[]
    for i in tqdm(range(len(src_corpus))):
        ner.append(nlp.ner(src_corpus[i]))
    
    saveNER(ner, src+'.ner')
    
    word2ner, ner2word = word2NER(ner)
    with open(src + '.ner_dict', 'wb') as f:
                 pickle.dump(ner2word, f)

    for i in range(len(src_corpus)):
        src_corpus[i] = replaceNER(src_corpus[i].split(), word2ner[i])
        tgt_corpus[i] = replaceNER(tgt_corpus[i].split(), word2ner[i])

    saveCorpus(src_corpus, src + '.ner_parse')
    saveCorpus(tgt_corpus, tgt + '.ner_parse')



if __name__ == '__main__':
    # dataRoot = '/home/linzhe/data/simplification/newsela/'
    # trainSrcOri = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.src'
    # validSrcOri = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.src'
    # testSrcOri  = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src'
    # trainTgtOri = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.dst'
    # validTgtOri = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.dst'
    # testTgtOri  = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.dst'
    
    # dataRoot = '/home/linzhe/data/simplification/wikismall/'
    # trainSrcOri = dataRoot + 'PWKP_108016.tag.80.aner.ori.train.src'
    # validSrcOri = dataRoot + 'PWKP_108016.tag.80.aner.ori.valid.src'
    # testSrcOri  = dataRoot + 'PWKP_108016.tag.80.aner.ori.test.src'
    # trainTgtOri = dataRoot + 'PWKP_108016.tag.80.aner.ori.train.dst'
    # validTgtOri = dataRoot + 'PWKP_108016.tag.80.aner.ori.valid.dst'
    # testTgtOri  = dataRoot + 'PWKP_108016.tag.80.aner.ori.test.dst'
    
    dataRoot    = '/home/linzhe/data/simplification/wikilarge/'
    trainSrcOri = dataRoot + 'wiki.full.aner.ori.train.src'
    validSrcOri = dataRoot + 'wiki.full.aner.ori.valid.src'
    testSrcOri  = dataRoot + 'wiki.full.aner.ori.test.src'
    trainTgtOri = dataRoot + 'wiki.full.aner.ori.train.dst'
    validTgtOri = dataRoot + 'wiki.full.aner.ori.valid.dst'
    testTgtOri  = dataRoot + 'wiki.full.aner.ori.test.dst'

    fileList = [(trainSrcOri, trainTgtOri), (validSrcOri, validTgtOri), (testSrcOri,  testTgtOri)]
    i = 1
    for file in fileList:
        print('%d / %d' % (i, len(fileList))) 
        i += 1
        NERPaser(file[0], file[1])
