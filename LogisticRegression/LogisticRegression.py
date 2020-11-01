import numpy as np
import matplotlib.pyplot as plt



def sigmoid(input):
    denominator = (np.exp(-1*input)) + 1
    return 1/denominator

def cost(weight, b, x, y):
    cost = 0
    predictedVals = [sigmoid((weight*i) + b) for i in x]
    for i in range(len(x)):
        #closer to zero means lower value from log
        cost+= ((y[i]*np.log(predictedVals[i])) + (1-y[i])*np.log(1 - predictedVals[i]))
    return (-cost)/len(x)

def Mgradient(weight, b, x, y):
    predictedVals = [sigmoid((weight*i) + b) for i in x]
    gradient = 0
    for i in range(len(x)):
        gradient += x[i] * (predictedVals[i] - y[i])
    return gradient/len(x)

def Bgradient(weight, b, x, y):
    predictedVals = [sigmoid((weight*i) + b) for i in x]
    gradientB = 0
    for i in range(len(x)):
        gradientB += (predictedVals[i] - y[i])
    return gradientB/len(x)


rand = (int)(np.random.random() * 50)
while(rand < 15):
    rand = (int)(np.random.random() * 50)

xList = [i for i in range(-15, 15, 1)]
yList = []
split = (int)(np.random.random() * 5)
while(abs(split) > 15):
    split = (int)(np.random.random() * 5)
    if split % 2 == 0:
        split *= -1

for i in xList:
    if i <= split:
        yList.append(0)
    else:
        yList.append(1)

#xCost = [i for i in range(2000)]
yCost = []

m = 0
b = 0
costVal = 1000.12
num = 0
while costVal > .15 or num < 5000:
    m -= Mgradient(m,b, xList, yList) * .001
    b -= Bgradient(m,b,xList,yList)*.001
    costVal = cost(m, b, xList, yList)
    num +=1
    #yCost.append(cost(m,b,xList,yList))






predictedVals = [sigmoid((m*i) + b) for i in xList]
print(cost(m,b,xList,yList))
print(m, b)
#plt.plot(xCost, yCost)
plt.scatter(xList, yList)
plt.plot(xList, predictedVals)


plt.show()

