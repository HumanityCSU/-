from os import error, terminal_size
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gradientDescent_and_computeCost import batch_gradientDescent
path = r'C:\Users\Alice\Desktop\吴恩达机器学习作业\data_sets\ex1data1.txt'


data = pd.read_csv(path,header=None,names=['Population','Profit'])
# print(data.head())
# print(data.describe())

def computeCost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))

data.insert(0,'ones',1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]# iloc函数有点像切片
y = data.iloc[:,cols-1:cols]
# print(X.head())

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

# print(X.shape,y.shape,theta.shape)
# cost = computeCost(X,y,theta)
# print(cost)

# 梯度下降函数
def gradientDescent(X,y,theta,alpha,iters): # inters:迭代次数
    temp = np.matrix(np.zeros(theta.shape))# 构建0值矩阵
    parameters = int(theta.ravel().shape[1])# ravel计算需要求解的个数 功能是将多维数组展平成1维
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T)-y
        for j in range(parameters):
            term = np.multiply(error,X[:,j])# 计算两个矩阵（h^θ(x)-y）*x
            temp[0,j] = theta[0,j]-((alpha/len(X))*np.sum(term))# 参数更新公式：θ(J) = θ(J)-α(∂(J)/∂(θ(j))) , ∂:偏微分符号,α:学习率
       
        theta = temp
        cost[i] = computeCost(X,y,theta)
    return theta,cost

alpha = 0.01
iters = 1000

# 梯度下降,或批量梯度下降
g,cost = batch_gradientDescent(X,y,theta,alpha,iters,batch=int(X.shape[1]/10))
print(g)
print(computeCost(X,y,g))

# 画图
x = np.linspace(data.Population.min(),data.Population.max(),100)# 抽100个样本
f = g[0,0]+g[0,1]*x # g[0,0]代表theta0,g[0,1]代表theta1
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Traning Data')
ax.legend(loc=4)# 显示标签位置
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
# plt.savefig('./一维线性回归拟合图')
plt.show()

