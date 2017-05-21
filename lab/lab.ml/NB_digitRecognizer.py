from numpy import *
import time

def csv2vector(filename,index=0):
    fr = open(filename)
    filelines = fr.readlines()
    del filelines[0]
    lenlines = len(filelines)
    print zeros(4)
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


def trainNB0(trainMatrix, trainclass):
    numpics = len(trainMatrix)   # record how many pictures
    numpix = len(trainMatrix[0]) # record how many pixels in one picture

    #cal the propability of each class
    pDicClass = {} 
    for v in trainclass:
        pDicClass[v] = pDicClass.get(v,0) + 1
    for k, v in pDicClass.items():
        pDicClass[k] = v/float(numpics) # p of every class

    #cal the every pixel sum of each class
    pDicNum = {}
    pDicNumSum = {}
    for k in pDicClass.keys():
        pDicNum[k] = ones(numpix)
    for i in range(numpics):
        pDicNum[trainclass[i]] += trainMatrix[i]
        pDicNumSum[trainclass[i]] = pDicNumSum.get(trainclass[i],2) + sum(trainMatrix[i])
    
    #cal the probability of every pixel of each class
    pDicNumVec = {}
    for k in pDicNum.keys():
        pDicNumVec[k] = log(pDicNum[k]/float(pDicNumSum[k]))

    return pDicNumVec, pDicClass
    

def classifyNB(vec2class, pDicNumVec, pDicClass):
    presult = {}
    for k in pDicClass.keys():
        presult[k] = sum(vec2class*pDicNumVec[k]) + log(pDicClass[k])

    tmp = float("-inf")
    result = ""
    for k in presult.keys():
        if presult[k] > tmp:
            tmp = presult[k]
            result = k

    return result


def testNB():
    print "load train data ..."
    trainSet, trainLabel = csv2vector("train.csv", 1)
    print trainLabel
    print "load test  data ..."
    testSet, testLabel = csv2vector("test.csv")
    print "start train ..."
    pDicNumVec, pDicClass = trainNB0(trainSet, trainLabel)
    start = time.clock()
    print "start test ..."
    result = "ImageId, Label\n"
    testloss = 0
    for i in range(len(testSet)):
        predictLabel = classifyNB(testSet[i], pDicNumVec, pDicClass)
        result += str(i+1) + "," + predictLabel+ "\n"
        if predictLabel is not testLabel[i]:
            testloss += 1
            #print i, predictLabel, testLabel[i], testloss
    testloss = float(testloss)/len(testSet)
    print testloss
    end = time.clock()
    print "time cost: %f s ..." % (end - start)
