import  torch
import  Constants
import  Parameter
from    Parameter import device
import  time
from    utils import splitLine, printInfo
from    checkpoint import saveModel, loadModel, saveOutput
from    eval import eval_score
from    numpy import inf
from    generate import generator
import  os
import  shutil

class fit:
    
    def __init__(self, *args, **kwargs):
        self.model            = kwargs['model']
        self.criterion        = kwargs['criterion']
        self.optimizer        = kwargs['optimizer']
        self.checkpoint       = kwargs['checkpoint'] if kwargs.get('checkpoint') is not None else None
        self.epoch            = kwargs['epoch'] if kwargs.get('epoch') is not None else 100
        self.numBatchPrint    = kwargs['numBatchPrint'] if kwargs.get('numBatchPrint') is not None else 100
        self.validDebug       = kwargs['validDebug'] if kwargs.get('validDebug') is not None else False
        self.testDebug        = kwargs['testDebug'] if kwargs.get('testDebug') is not None else False
        self.debugFile        = kwargs['debugFile'] if kwargs.get('debugFile') is not None else './log/'
        self.maxlen           = kwargs['maxlen'] if kwargs.get('maxlen') is not None else 1000
        self.gradient_clipper = kwargs['gradient_clipper'] if kwargs.get('gradient_clipper') is not None else 1000

        self.trainData = None
        self.testData = None
        self.validData = None
        self.index2word = None

    def run(self, Data, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        ntokens = 0
        total_loss = 0
        total_seq = 0
        batch_st = time.time()
        for i, (source, graph, rgraph, tgtInputs, tgtOutputs) in enumerate(Data):
            source = source.to(device)
            tgtInputs = tgtInputs.to(device)
            tgtOutputs = tgtOutputs.to(device)
            graph = graph.to(device)
            rgraph = rgraph.to(device)

            length = (tgtInputs != Constants.PAD).sum(-1)
            lentok = torch.LongTensor(length.size(0)).fill_(Constants.MIDDLE).to(device)
            lentok.masked_fill_(length < Parameter.t_min, Constants.SHORT)
            lentok.masked_fill_(length > Parameter.t_max, Constants.LONG)
            source = torch.cat((lentok.unsqueeze(-1), source), dim=-1)


            outputs, penalty = self.model(source, tgtInputs, graph, rgraph, Constants.BOS, Constants.EOS, Constants.PAD)
            loss = self.criterion(outputs, tgtOutputs, penalty, Constants.PAD)

            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipper)
                self.optimizer.step()
            ntokens += (source != Constants.PAD).sum().item()
            total_loss += loss.item() * source.size(0)
            total_seq += source.size(0)

            if mode == 'train' and i % self.numBatchPrint == 0:
                total_time = time.time() - batch_st
                print('BATCH: %4d\t\tloss: %.4f\t\tTime: %4d\tTokens per Sec: %d' 
                        %(i, total_loss/total_seq, total_time, ntokens / total_time))
                batch_st = time.time()
                ntokens = 0
                total_loss = 0
                total_seq = 0

        if mode != 'train':
            return total_loss / total_seq

    def train(self):

        _epoch = 0; bestLoss = inf
        if self.checkpoint is not None:
            self.model, self.optimizer, _epoch, bestLoss \
                                = self.checkpoint.restoreCheckPoint(self.model, self.optimizer)

        for epoch in range(_epoch, self.epoch):
            epoch_st = time.time()
            splitLine('+', 85)
            print('EPOCH: %d' % (epoch + 1))
            splitLine('-', 85)
            self.run(self.trainData, mode='train')
            with torch.no_grad():        
                loss = self.run(self.validData, mode='valid')
            if self.testDebug or self.validDebug:
                self.debug(epoch)
            if epoch == 0 or bestLoss > loss:
                bestLoss = loss
                saveModel(self.model, Parameter.modelPath, Parameter.modelFile)        

            splitLine('-', 85)
            print('Valid Loss: %.4f\t\tBest Valid Loss: %.4f\t\tTime: %d' % (loss, bestLoss, time.time()-epoch_st))
            self.checkpoint.saveCheckPoint(self.model, self.optimizer, epoch + 1, bestLoss)
        self.model.load_state_dict(loadModel(Parameter.modelPath, Parameter.modelFile))

    def debug(self, epoch):
        if epoch == 0:
            if os.path.exists(self.debugFile):
                shutil.rmtree(self.debugFile)
        epoch += 1
        if os.path.exists(self.debugFile) == False:
            os.mkdir(self.debugFile)
        if os.path.exists(self.debugFile + 'epoch_%d' % epoch) == False:
            os.mkdir(self.debugFile + 'epoch_%d' % epoch)

        saveModel(self.model, self.debugFile + 'epoch_%d/' % epoch, 'model.pth')        
        if self.validDebug:
            validOut = generator(self.model, self.validData, Parameter.validSrcFilePath, self.index2word,
                                      Parameter.validDictPath, Parameter.search_method, Parameter.beam_size)
            saveOutput(validOut, self.debugFile + 'epoch_%d/' % epoch, 'valid')
            bleu_valid, sari_valid, fkgl_valid = eval_score(Parameter.validSrcOri, Parameter.validTgtOrih, 
                                                            self.debugFile + 'epoch_%d/valid' % epoch)
        if self.testDebug:
            testOut = generator(self.model, self.testData, Parameter.testSrcFilePath, self.index2word,
                                     Parameter.testDictPath, Parameter.search_method, Parameter.beam_size)
            saveOutput(testOut, self.debugFile + 'epoch_%d/' % epoch, 'test')
            bleu_test, sari_test, fkgl_test = eval_score(Parameter.testSrcOri, Parameter.testTgtOri, 
                                                            self.debugFile + 'epoch_%d/test' % epoch)
        with open(self.debugFile + 'score.txt', 'a') as f:
            f.write('EPOCH: %d\n' % epoch)
            if self.validDebug:
                f.write('valid:\tBLEU: %.2f\tSARI: %.2f\tFKGL: %.2f\n' % (bleu_valid, sari_valid, fkgl_valid))
            if self.testDebug:
                f.write('test:\tBLEU: %.2f\tSARI: %.2f\tFKGL: %.2f\n' % (bleu_test, sari_test, fkgl_test))
            f.write('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

    def __call__(self, trainData, validData, testData, index2word):
        self.trainData = trainData
        self.testData = testData
        self.validData = validData
        self.index2word = index2word
        self.train()