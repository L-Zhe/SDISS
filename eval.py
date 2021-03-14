from    nltk.translate import bleu_score
from    Readability.readability import Readability 
import  numpy as np
import  os
import  subprocess

os.environ['JAVA_HOME'] = '/envs/remote/jdk-1.8/bin/java'
os.environ['JOSHUA']    = '/home/linzhe/tool/SARI/ppdb-simplification-release-joshua5.0/joshua'
os.environ['LC_ALL']    = 'en_US.UTF-8'
os.environ['LANG']      = 'en_US.UTF-8'
home = '/home/linzhe/tool/SARI/ppdb-simplification-release-joshua5.0/joshua/bin'

def corpus_bleu(candidate, reference):
    return bleu_score.corpus_bleu(reference, candidate,
                    smoothing_function=bleu_score.SmoothingFunction().method3) * 100

def corpus_fkgl(output):
    text = '\n'.join(output)
    try:
        FKGL=Readability(text).FleschKincaidGradeLevel()
    except ZeroDivisionError:
        FKGL=-np.inf
    return FKGL

def corpus_sari(src_path, tgt_path, gen_path):
    cmd = 'bash ' + home + '/star_1'+ ' ' + gen_path +' ' + tgt_path + ' ' + src_path
    output = os.popen(cmd)
    sari = output.read().split('\n')[-3].split()[-1]
    return float(sari) * 100

def corpus_sari8(src_path, tgt_path, gen_path):
    cmd = 'bash ' + home + '/star'+ ' ' + gen_path +' ' + tgt_path + ' ' + src_path
    output = os.popen(cmd)
    sari = output.read().split('\n')[-3].split()[-1]
    return float(sari) * 100

def readFile(file, ref=False):
    with open(file, 'r') as f:
        if ref:
            corpus = [[[word.lower() for word in line.split()]] for line in f.readlines()]
        else:
            corpus = [[word.lower() for word in line.split()] for line in f.readlines()]
    return corpus

def eval_score(srcFile, refFile, canFile, savePath=None):
    source = readFile(srcFile)
    reference = readFile(refFile, ref=True)
    candidate = readFile(canFile)
    bleu = corpus_bleu(candidate, reference)
    sari = corpus_sari(srcFile, refFile, canFile)
    fkgl = corpus_fkgl([' '.join(line) for line in candidate])
    if savePath is not None:
        with open(savePath + 'score.txt', 'w') as f:
            f.write('BLEU: %.2f\tSARI: %.2f\tFKGL: %.2f' % (bleu, sari, fkgl))
    print('BLEU: %.2f\tSARI: %.2f\tFKGL: %.2f' % (bleu, sari, fkgl))
    return bleu, sari, fkgl

if __name__ == '__main__':
    import  Parameter
    # eval_score('/home/linzhe/1.src', '/home/linzhe/1.tgt', 
            #    '/home/linzhe/Project/wikismall_editnts.txt')
    # sysdir= '/home/linzhe/Project/test/gcn_copy_wikilarge_lentok_copyloss/output/log/epoch_0/test'
    sysdir = '/home/linzhe/Project/SemSS/wikilarge/log/epoch_1/test'
    input = '/home/linzhe/simplification/data/turkcorpus/test.8turkers.tok.norm'
    ref   = '/home/linzhe/simplification/data/turkcorpus/test.8turkers.tok.turk'
    print(corpus_sari8(input, ref, sysdir))
