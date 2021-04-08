import  torch
from    torch import nn
from    torch import optim
from    torch.utils.data import DataLoader
import  Parameter
from    Parameter import device
from    model import createModel
from    SDISS.Optim import WarmUpOpt, LabelSmoothing
from    checkpoint import CheckPoint, saveOutput, loadModel
from    run import fit
from    eval import eval_score
from    generate import generator
import  os

if __name__ == '__main__':
    checkpoint = CheckPoint(trainSrcFilePath = Parameter.trainSrcFilePath,
                            validSrcFilePath = Parameter.validSrcFilePath,
                            testSrcFilePath  = Parameter.testSrcFilePath,
                            trainTgtFilePath = Parameter.trainTgtFilePath,
                            validTgtFilePath = Parameter.validTgtFilePath,
                            testTgtFilePath  = Parameter.testTgtFilePath,
                            trainGraph       = Parameter.trainGraph,
                            validGraph       = Parameter.validGraph,
                            testGraph        = Parameter.testGraph,
                            min_freq         = Parameter.min_freq,
                            BATCH_SIZE       = Parameter.BATCH_SIZE,
                            dataPath         = Parameter.dataPath,
                            dataFile         = Parameter.dataFile,
                            checkpointPath   = Parameter.checkpointPath,
                            checkpointFile   = Parameter.checkpointFile,
                            mode             = Parameter.mode)

    trainDataSet, validDataSet, testDataSet, index2word, gword2index = checkpoint.LoadData()   
    model = createModel(len(index2word), len(gword2index)).to(device)
    if Parameter.mode == 'train':
        criterion = LabelSmoothing(smoothing=Parameter.smoothing, lamda=Parameter.lamda).to(device)

        optimizer = WarmUpOpt(optimizer    = optim.Adam(params=model.parameters(), 
                                                        lr=Parameter.learning_rate,
                                                        betas=(Parameter.beta_1, Parameter.beta_2),
                                                        eps=Parameter.eps,
                                                        weight_decay=Parameter.weight_decay),
                              d_model      = Parameter.embedding_dim,
                              warmup_steps = Parameter.warmup_steps,
                              factor       = Parameter.factor)

        fit = fit(model            = model,
                  criterion        = criterion,
                  optimizer        = optimizer,
                  checkpoint       = checkpoint,
                  epoch            = Parameter.EPOCH,
                  numBatchPrint    = Parameter.numBatchPrint,
                  validDebug       = Parameter.validDebug,
                  testDebug        = Parameter.testDebug,
                  debugFile        = Parameter.debugFile,
                  maxlen           = Parameter.maxLen,
                  gradient_clipper = Parameter.gradient_clipper)

        fit(trainDataSet, validDataSet, testDataSet, index2word)
    else:
        model.load_state_dict(loadModel(Parameter.modelPath, Parameter.modelFile))
        model.cuda()
        candidate = generator(model, testDataSet, Parameter.testSrcFilePath, index2word,
                              Parameter.testDictPath, Parameter.search_method, Parameter.beam_size)
        saveOutput(candidate, Parameter.outputPath, Parameter.outputFile)
        eval_score(Parameter.testSrcOri, Parameter.testTgtOri, 
                   Parameter.outputPath + Parameter.outputFile,
                   Parameter.outputPath)