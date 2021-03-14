import  torch
from    torch.utils.data import DataLoader
from    Preprocess import Vocab, lang, graph, graphVocab, get_dataloader
import  os
from    numpy import inf

class CheckPoint:
    def __init__(self, *args, **kwargs):
        self.trainSrcFilePath = kwargs['trainSrcFilePath']
        self.validSrcFilePath = kwargs['validSrcFilePath']
        self.testSrcFilePath  = kwargs['testSrcFilePath'] 
        self.trainTgtFilePath = kwargs['trainTgtFilePath']
        self.validTgtFilePath = kwargs['validTgtFilePath']
        self.testTgtFilePath  = kwargs['testTgtFilePath'] 
        self.trainGraph       = kwargs['trainGraph']
        self.validGraph       = kwargs['validGraph']
        self.testGraph        = kwargs['testGraph'] 
        self.min_freq         = kwargs['min_freq'] 
        self.BATCH_SIZE       = kwargs['BATCH_SIZE'] 

        self.dataPath       = kwargs['dataPath']
        self.dataFile       = kwargs['dataFile']
        self.checkpointPath = kwargs['checkpointPath']
        self.checkpointFile = kwargs['checkpointFile']
        self.mode           = kwargs['mode']

        self.loadDataSuccess = False
        if os.path.exists(self.dataPath) == False:
            os.makedirs(self.dataPath)
        if os.path.exists(self.checkpointPath) == False:
            os.makedirs(self.checkpointPath)

    def LoadData(self):
        try:
            data = torch.load(self.dataPath + self.dataFile)
            word2index = data['word2index']
            index2word = data['index2word']
            gword2index = data['gword2index']
            self.loadDataSuccess = True

        except:
            if self.mode == 'train':
                word2index, index2word = Vocab([self.trainSrcFilePath, self.trainTgtFilePath], self.min_freq)
                gword2index = graphVocab(self.trainGraph, 2)
                data = {}
                data['word2index']  = word2index 
                data['index2word']  = index2word 
                data['gword2index'] = gword2index
                torch.save(data, self.dataPath + self.dataFile)
            else:
                raise IOError("Can not find valid vocabulary file...")
        if self.mode == 'train':
            srcTrain = lang(self.trainSrcFilePath, word2index, PAD=True)
            tgtTrainInputs = lang(self.trainTgtFilePath, word2index, BOS=True, PAD=True)
            tgtTrainOutputs = lang(self.trainTgtFilePath, word2index, EOS=True, PAD=True)
            srcValid = lang(self.validSrcFilePath, word2index, PAD=True)
            tgtValidInputs = lang(self.validTgtFilePath, word2index, BOS=True, PAD=True)
            tgtValidOutputs = lang(self.validTgtFilePath, word2index, EOS=True, PAD=True)
            trainLen = len(srcTrain[0])
            validLen = len(srcValid[0])
            trainGraph, rtrainGraph = graph(self.trainGraph, gword2index, trainLen)
            validGraph, rvalidGraph = graph(self.validGraph, gword2index, validLen)
            trainDataSet = get_dataloader(srcTrain, trainGraph, rtrainGraph, tgtTrainInputs, tgtTrainOutputs,
                                          batch_size=self.BATCH_SIZE, shuffle=True)
            validDataSet = get_dataloader(srcValid, validGraph, rvalidGraph, tgtValidInputs, tgtValidOutputs,
                                          batch_size=self.BATCH_SIZE, shuffle=False)
            testDataSet = None
        else:
            srcTest = lang(self.testSrcFilePath, word2index, PAD=True)
            testLen  = len(srcTest[0])
            testGraph, rtestGraph  = graph(self.testGraph, gword2index, testLen)
            trainDataSet = None
            validDataSet = None
            testDataSet = get_dataloader(srcTest, testGraph, rtestGraph, batch_size=self.BATCH_SIZE, shuffle=False)

        return trainDataSet, validDataSet, testDataSet, index2word, gword2index

    def saveCheckPoint(self, model, optim, epoch, bestLoss):
            checkpoint = {'model': model.state_dict(),
                          'optim': optim,
                          'epoch': epoch,
                          'bestLoss': bestLoss}
            torch.save(checkpoint, self.checkpointPath + self.checkpointFile)
            
    def restoreCheckPoint(self, model, optimizer):
        if self.loadDataSuccess:
            try:
                checkpoint = torch.load(self.checkpointPath + self.checkpointFile)
                model.load_state_dict(checkpoint['model'])
                return model, checkpoint['optim'], checkpoint['epoch'], checkpoint['bestLoss']
            except IOError:
                return model, optimizer, 0, inf
        else:
            return model, optimizer, 0, inf

def saveModel(model, modelPath, modelFile):
    if os.path.exists(modelPath) == False:
        os.makedirs(modelPath)
    torch.save(model.state_dict(), modelPath + modelFile)

def loadModel(modelPath, modelFile):
    return torch.load(modelPath + modelFile)

def saveOutput(candidate, path, file):
    if os.path.exists(path) == False:
        os.makedirs(path)
    with open(path + file, 'w') as f:
        for line in candidate:
            f.write(' '.join(line))
            f.write('\n')

