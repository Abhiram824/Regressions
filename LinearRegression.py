import numpy as np
import matplotlib.pyplot as plt

def cost(listy, listx, m, b):
  predictedVals = [(m * x) + b for x in listx]
  sumCost = 0
  for i in range(len(predictedVals)):
    sumCost += (y[i] - predictedVals[i]) ** 2
  return sumCost/len(predictedVals)

def bGradient(listy, listx, m, b):
  predictedVals = [(m * x) + b for x in listx]
  bGradient = 0
  for i in range(len(predictedVals)):
    bGradient += ((listy[i] - predictedVals[i]) * -2)
    #partial derivative with respect to intercept
  return bGradient/len(predictedVals)

def mGradient(listy, listx, m, b):
   predictedVals = [(m * x) + b for x in listx]
   mGradient = 0
   for i in range(len(predictedVals)):
    mGradient += ((listy[i] - predictedVals[i]) * -2) * listx[i] 
    #partial derivative with respect to slope
   return mGradient/len(predictedVals)


  



listx = np.linspace(0, 9, 10)
y = [np.random.random()*2-1 + i for i in listx]
m = 0
b = 0
for i in range(100):
  learningRate = .001
  m -= mGradient(y, listx, m, b) * learningRate
  b -= bGradient(y,listx,m,b) * learningRate

predictedVals = [(m * x) + b for x in listx]


dataset = np.array([listx, y])
plt.plot(listx, predictedVals)
plt.scatter(listx, y)

plt.show()
# print(dataset)
print(cost(listx, y, m, b)) 

d1 = np.array(y) - np.array(predictedVals)
d2 = np.array(y) - np.array(y).mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print('r^2 = ' + str(r2))