import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.metrics import confusion_matrix,f1_score,roc_curve,auc,precision_recall_curve,accuracy_score
from sklearn.model_selection import train_test_split,KFold,LeavePOut,LeavePGroupsOut
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn import preprocessing
import sklearn.linear_model as LM
from sklearn import neighbors

#模型复杂度和训练函数的关系
np.random.seed(123)
N = 200
x = np.linspace(0.1,10,num=N)
y = []
z = []
for i in range(N):
    tmp = 10*np.math.sin(4*x[i])+10*x[i]+20*np.math.log(x[i])+30*np.math.cos(x[i])
    y.append(tmp)
    tmp = y[i]+np.random.normal(0,3)
    z.append(tmp)

fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(15,4))
axes[0].scatter(x,z,s=5)
axes[0].plot(x,y,'k-',label='真实关系')

modelLR = LM.LinearRegression()
X = x.reshape(N,1)
Y = np.array(z)
modelLR.fit(X,Y)
axes[0].plot(x,modelLR.predict(X),label='线性模型')
linestyle = ['--','-.',':','-']
for i in np.arange(1,5):
    tmp = pow(x,(i+1)).reshape(N,1)
    X = np.hstack((X,tmp))
    modelLR.fit(X,Y)
    axes[0].plot(x,modelLR.predict(X),linestyle=linestyle[i-1],label=str(i+1)+'项式')
axes[0].legend()
axes[0].set_title('真实关系和不同复杂度模型的拟合情况')
axes[0].set_xlabel('输入变量')
axes[0].set_ylabel('输出变量')
# plt.show()
# plt.savefig(r'./模型复杂度与误差的模拟研究.png')

#计算各个模型的训练误差
X = x.reshape(N,1)
Y = np.array(z)
modelLR.fit(X,Y)
MSEtrain = [np.sum((Y-modelLR.predict(X)))**2/(N-2)]
for i in np.arange(1,9):
    tmp = pow(x,(i+1)).reshape(N,1)
    X = np.hstack((X,tmp))
    modelLR.fit(X,Y)
    MSEtrain.append(np.sum((Y-modelLR.predict(X))**2)/(N-(i+2)))
axes[1].plot(np.arange(1,10),MSEtrain,marker='o',label='训练误差')
axes[1].legend()
axes[1].set_title('不同复杂度模型的训练误差')
axes[1].set_xlabel('模型复杂度')
axes[1].set_ylabel('MSE')

#研究样本量对训练误差的影响
X = x.reshape(N,1)
Y = np.array(z)
for i in np.arange(1,5):
    tmp = pow(x,(i+1)).reshape(N,1)
    X = np.hstack((X,tmp))
np.random.seed(0)
size = np.linspace(0.2,0.99,100)
MSEtrain = []
for i in range(len(size)):
    Ntraining = int(N*size[i])
    id = np.random.choice(N,Ntraining,replace=False)
    X_train = X[id]
    Y_train = Y[id]
    modelLR.fit(X_train,Y_train)
    MSEtrain.append(np.sum((Y_train-modelLR.predict(X_train))**2)/(Ntraining-6))

tmpx = np.linspace(1,len(size),num=len(size)).reshape(len(size),1)
tmpX = np.hstack((tmpx,tmpx**2))
tmpX = np.hstack((tmpX,tmpx**3))
modelLR.fit(tmpX,MSEtrain)

axes[2].plot(size,MSEtrain,linewidth=1.5,linestyle='-',label='MSE')
axes[2].plot(size,modelLR.predict(tmpX),linewidth=1.5,linestyle='--',label='MSE趋势线')
axes[2].set_title('训练样本量和训练MSE')
axes[2].set_xlabel('训练样本水平')
axes[2].set_ylabel('MSE')
axes[2].legend()
plt.savefig(r'./模型复杂度与误差的模拟研究.png')
plt.show()


