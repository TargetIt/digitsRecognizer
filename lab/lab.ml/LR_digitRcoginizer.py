from numpy import *
import time
import threading

def csv2vector(filename,index=0):
    fr = open(filename)
    filelines = fr.readlines()
    del filelines[0]
    lenlines = len(filelines)
    returnVect = zeros((lenlines,784))
    labellist = [0]*lenlines
    for i in range(lenlines):
        lineStr = filelines[i]
        linearr = lineStr.split(',')
        if len(linearr)< 784:
            continue
        labellist[i] = linearr[0]
        for j in range(index,len(linearr)):
            if linearr[j] != '0':
                returnVect[i,j-index] = 1
            else:
                returnVect[i,j-index] = 0
    fr.close()
    return returnVect,labellist

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels, numIter=500):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    weights = ones((n,1))
    for k in range(numIter):
        h = sigmoid(dataMatrix*weights) # Matrix multiplication
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=100):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def stocGradAscent1_qp(dataMatrix, classLabels, numIter=100):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            usedIndex = dataIndex[randIndex]
            h = sigmoid(sum(dataMatrix[usedIndex] * weights))
            error = classLabels[usedIndex] - h
            weights = weights + alpha * error * dataMatrix[usedIndex]
            del(dataIndex[randIndex])
    return weights

## ------------------------------------------------------------------
## output : one-hot encoding, ten output node totally
## only train for one node each thread, there are 10 threads totally
## ------------------------------------------------------------------

g_weights_list = [0,0,0,0,0,0,0,0,0,0]

def trainweight(data, labels, tag):
    we = getweight(data, labels, str(tag))
    g_weights_list[tag] = we
    return g_weights_list

# only CONV the current tag label to 1, others to 0
def getlabels (labels, tag):
    labels_ = list(labels)
    for i in range(len(labels_)):
        if labels_[i] == tag:
            labels_[i] = 1
        else:
            labels_[i] = 0
    return labels_

def getweight(testdata, testlabels, tag):
    labels = getlabels(testlabels, tag)
    we = stocGradAscent1_qp(testdata, labels)
    we = mat(we).transpose()
    return we

def train(dataMatrix, classLabels):
    print "train start ..."
    start = time.clock()
    threads = []
    for i in range(10):
        t = threading.Thread(target=trainweight, args=(dataMatrix, classLabels, i))
        threads.append(t)
    for i in range(len(threads)):
        threads[i].start()
        print "thread", i, " start"
    for i in range(len(threads)):
        threads[i].join()
        print "thread", i, " end"
    print "train end  ..."
    end = time.clock()
    print "train time cost: %f s" % (end - start)
    return g_weights_list

def classify(inX, weights_list):
    maxProb = 0
    labels = -1
    for i in range(len(weights_list)):
        curProb = sigmod(inX * weights_list[i])
        if curPorb > maxProb:
            maxProb = curProb
            labels = i
    return labels


def test(testData, testLabel, we_list):
    error = 0
    for i in range(len(testData)):
        rec = classify(testData[i], we_list)
        if testlabel[i] != str(rec):
            error = error + 1
            print testlabel[i], ",", rec

def test_LR():
    print "load train data ..."
    trainSet, trainLabel = csv2vector("train.csv", 1)
    print "load test data ..."
    testSet, testLabel = csv2vector("test.csv")
    print "start train ..."
    g_weights_list = train(trainSet, trainLabel)
    for i in range(len(testSet)):
        test(testSet[i], testLabel[i], g_weights_list)

