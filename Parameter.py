import  os
import  torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
USE_CUDA   = torch.cuda.is_available()
device     = torch.device("cuda:0" if USE_CUDA else "cpu")
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
data_set = 'newsela'
if data_set == 'newsela':
    dataRoot    = '/home/linzhe/data/simplification/newsela/'
    trainSrcOri = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.src'
    validSrcOri = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.src'
    testSrcOri  = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src'
    trainTgtOri = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.dst'
    validTgtOri = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.dst'
    testTgtOri  = dataRoot + 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.dst'
    trainGraph  = dataRoot +'sdp/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.src.sdp'
    validGraph  = dataRoot +'sdp/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.src.sdp'
    testGraph   = dataRoot + 'sdp/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src.sdp'
    lamda       = 5
    dataRoot    = './newsela/'
    numBatchPrint = 500
    BATCH_SIZE  = 50
    t_min       = 0
    t_max       = 15

elif data_set == 'wikismall':
    dataRoot    = '/home/linzhe/data/simplification/wikismall/'
    trainSrcOri = dataRoot + 'PWKP_108016.tag.80.aner.ori.train.src'
    validSrcOri = dataRoot + 'PWKP_108016.tag.80.aner.ori.valid.src'
    testSrcOri  = dataRoot + 'PWKP_108016.tag.80.aner.ori.test.src'
    trainTgtOri = dataRoot + 'PWKP_108016.tag.80.aner.ori.train.dst'
    validTgtOri = dataRoot + 'PWKP_108016.tag.80.aner.ori.valid.dst'
    testTgtOri  = dataRoot + 'PWKP_108016.tag.80.aner.ori.test.dst'
    trainGraph  = dataRoot +'sdp/PWKP_108016.tag.80.aner.ori.train.src.sdp'
    validGraph  = dataRoot +'sdp/PWKP_108016.tag.80.aner.ori.valid.src.sdp'
    testGraph   = dataRoot + 'sdp/PWKP_108016.tag.80.aner.ori.test.src.sdp'
    lamda       = 10
    dataRoot    = './wikismall/'
    numBatchPrint = 500
    BATCH_SIZE  = 40
    t_min       = 0
    t_max       = 20

else:
    dataRoot    = '/home/linzhe/data/simplification/wikilarge/'
    trainSrcOri = dataRoot + 'wiki.full.aner.ori.train.src'
    validSrcOri = dataRoot + 'wiki.full.aner.ori.valid.src'
    testSrcOri  = dataRoot + 'wiki.full.aner.ori.test.src'
    trainTgtOri = dataRoot + 'wiki.full.aner.ori.train.dst'
    validTgtOri = dataRoot + 'wiki.full.aner.ori.valid.dst'
    testTgtOri  = dataRoot + 'wiki.full.aner.ori.test.dst'
    trainGraph  = dataRoot +'sdp/wiki.full.aner.ori.train.src.sdp'
    validGraph  = dataRoot +'sdp/wiki.full.aner.ori.valid.src.sdp'
    testGraph   = dataRoot + 'sdp/wiki.full.aner.ori.test.src.sdp'
    lamda       = 10
    dataRoot    = './wikilarge/'
    numBatchPrint = 3000
    BATCH_SIZE  = 28
    t_min       = 5
    t_max       = 50

trainSrcFilePath = trainSrcOri + '.ner_parse'
trainTgtFilePath = trainTgtOri + '.ner_parse'
validSrcFilePath = validSrcOri + '.ner_parse'
validTgtFilePath = validTgtOri + '.ner_parse'
testSrcFilePath  = testSrcOri  + '.ner_parse'
testTgtFilePath  = testTgtOri  + '.ner_parse'

srcDictPath   = trainSrcOri + '.ner_dict'
validDictPath = validSrcOri + '.ner_dict'
testDictPath  = testSrcOri + '.ner_dict'

dataPath       =  './output/Vocab/';     dataFile       = 'Newsela_vocab.pkl'
checkpointPath = './checkpoint/';        checkpointFile = 'checkpoint.pth'
modelPath      = './output/Model/';      modelFile = 'Newsela_model.pth'
outputPath     =  './output/result/';    outputFile = 'result.txt'
debugFile      =  './log/'

mode          = 'test'
min_freq      = 2
EPOCH         = 30
maxLen        = 100
validDebug    = False
testDebug     = True

embedding_dim     = 256
d_ff              = 1024
num_head          = 4
num_head_gcn      = 4
num_layer_encoder = 6
num_layer_decoder = 6
num_layer_gcn     = 6
dropout_embed     = 0.1
dropout_sublayer  = 0.1
dropout_ffn       = 0.

search_method = 'beam'
beam_size     = 5
smoothing     = 0.1
warmup_steps  = 4000
factor        = 1

learning_rate    = 1e-4
beta_1           = 0.9
beta_2           = 0.98
eps              = 1e-9
weight_decay     = 1e-4
gradient_clipper = 5
