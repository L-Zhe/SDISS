def printInfo(EPOCH, BATCH_SIZE, max_len, USE_CUDA=False):

    print("++++++++++++++++++++++++++")
    print("Parameter INFO:")
    print("--------------------------")
    print("EPOCH: \t %d" % EPOCH)
    print("BATCH_SIZE: %d " % BATCH_SIZE)
    print("Generation Max Len: % d" % max_len)
    print("Device: ", "USE_CUDA" if USE_CUDA else "USE_CPU")
    print("++++++++++++++++++++++++++")

def splitLine(mode, num=64):
    if mode == '-':
        print('-' * num)
    elif mode == '+':
        print('+' * num)