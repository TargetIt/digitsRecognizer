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
	

def train(data,labels):
    print "train start"
    start = time.clock()
    threads = []
    for i in range(10):#开启10个线程计算
        t = threading.Thread(target=trainweight,args=(data,labels,i))
        threads.append(t)
    for i in range(len(threads)):
        threads[i].start()
        print "thread",i," start"
    for i in range(len(threads)):
        threads[i].join()
        print "thread",i," end"
    print "train end"
    end = time.clock()
    print "train time cost: %f s" % (end - start)
    return g_we_list

g_we_list=[0,0,0,0,0,0,0,0,0,0]
def trainweight(data,labels,tag):
    we = getweight(data,labels,str(tag))
    g_we_list[tag]=we

def getlabels(labels,tag):
    labels_ = list(labels)
    for i in range(len(labels_)):
        if labels_[i] ==tag:
            labels_[i]=1
        else:
            labels_[i]=0

    return labels_
def getweight(testdata,testlabels,tag):
    labels=getlabels(testlabels,tag)
    we = stocalcgrand1(testdata,labels)
    we = mat(we).transpose()
    return we

def stocalcgrand1(dataMatin,labelMatin,numiter=100):
    m,n=shape(dataMatin)
    alpha=0.01
    weight=ones(n)
    for i in range(numiter):
        dataIndex=range(m)
        for j in range(m):
            alpha=0.005/(1.0+i)+0.005
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmod(sum(dataMatin[randIndex]*weight))
            error=labelMatin[dataIndex[randIndex]]-h
            weight=weight+alpha*error*dataMatin[randIndex]
            del(dataIndex[randIndex])
    return weight

def test(testData,testlabel,we_list):
    error = 0
    for i in range(len(testData)):
        #for j in range(len(we_list)):
        rec = classfy(testData[i],we_list)
        if testlabel[i] != str(rec):
            error = error+1
            print testlabel[i],",",rec

    print "error=",error
    print "error percent:%f" % (float(error)/len(testData))
def classfy(testData,we_list):
    tmp=0
    labels = -1
    for i in range(len(we_list)):#根据sigmod的值大小来确定是哪一类
        sg = sigmod(testData*we_list[i])
        if sg>tmp:
            tmp = sg
            labels = i
    return labels