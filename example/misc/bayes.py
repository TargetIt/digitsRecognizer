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

def trainNB0(trainMatrix,trainclass):
    numpics = len(trainMatrix)  #record numbers
    numpix = len(trainMatrix[0])#pix numbers
    pDic={}
    for v in trainclass:
        pDic[v] = pDic.get(v,0)+1
    for k,v in pDic.items():
        pDic[k]=v/float(numpics)#p of every class
    pnumdic={}    
    psumdic={}
    for k in pDic.keys():
        pnumdic[k]=ones(numpix)
    for i in range(numpics):
        pnumdic[trainclass[i]] += trainMatrix[i]
        psumdic[trainclass[i]] = psumdic.get(trainclass[i],2) + sum(trainMatrix[i])
    pvecdic={}
    for k in pnumdic.keys():
        pvecdic[k]=log(pnumdic[k]/float(psumdic[k]))
    return pvecdic,pDic

def classifyNB(vec2class,pvecdic,pDic):
    presult={}    
    for k in pDic.keys():
        presult[k]=sum(vec2class*pvecdic[k])+log(pDic[k])
    tmp=float("-inf")
    result=""
    for k in presult.keys():
        if presult[k]>tmp:
            tmp= presult[k]
            result=k
    return result

def testNB():
    print "load train data..."
    trainSet, trainlabel=csv2vector("train.csv",1)
    print "load test data..."
    testSet,testlabel = csv2vector("test.csv")
    print "start train..."
    pvecdic,pDic=trainNB0(trainSet, trainlabel)
    start = time.clock()
    print "start test..."
    result="ImageId,Label\n"
    for i in range(len(testSet)):
        tmp = classifyNB(testSet[i],pvecdic,pDic)
        result += str(i+1)+","+tmp+"\n"
        #print tmp
    #savefile(result,"result_NB.csv")
    end = time.clock()
    print "time cost: %f s" % (end - start)
