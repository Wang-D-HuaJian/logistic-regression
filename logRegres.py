from numpy import *
###回归梯度上升优化算法
def loadDataSet():
    """读取文本信息并处理"""
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    """sigmoid函数"""
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """梯度上升算法"""
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix(100*3)
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix(100*1)
    #print(dataMatrix)
    #print(classLabels)
    #print(labelMat)
    m,n = shape(dataMatrix)
    #print(m,n)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))#########将回归系数初始化为1(3*1)
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult         (100*1)
        error = (labelMat - h)              #vector subtraction  (100*1)
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    #print(h)
    #print(error)
    return weights

def plotBestFit(weights):
    """画出数据集和Logistic回归最佳拟合直线"""
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    ##########对数据按标签进行分类，再按x,y值进行存储
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    #绘制最佳拟合直线
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    print(x)
    print(y)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    """随机梯度上升算法，随机选择一个样本点来更新回归系数，减少计算复杂度"""
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights



def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """随机梯度上升算法2.0"""
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #alpha每次迭代都会调整(变小 )，缓解数据波动和高频波动
            randIndex = int(random.uniform(0,len(dataIndex)))#随机取样进行更新回归系数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


##################################Logistic回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        #print("未能存活(1.0)")
        return 1.0
    else:
        #print("仍存活(0.0)")
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 200)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

multiTest()