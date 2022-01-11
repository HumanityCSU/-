from typing import final
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gradientDescent_and_computeCost import computeCost, gradientDescent

# 读取文件
path = r'C:\Users\Alice\Desktop\吴恩达机器学习作业\data_sets\ex1data2.txt'
data2 = pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])
# print(data2.head())

# 归一化,因为如果数据大小相差太大不好收敛
data2 = (data2-data2.mean())/data2.std()
# print(data2.head())

# 插入全1列
data2.insert(0,'Ones',1)

# 设置训练集
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# 变换为矩阵格式和初始化参数theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

#设置学习率和迭代次数
alpha = 0.01
iters = 1000

# 在训练集做梯度下降
g2,cost2 = gradientDescent(X2,y2,theta2,alpha,iters)


print(computeCost(X2,y2,g2))

# 画Cost图
fig,ax = plt.subplots(figsize = (12,8))
ax.plot(np.arange(iters),cost2,'r')
ax.set_xlabel('Interations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Traning Epoch')
# plt.savefig('./多维线性回归的Cost随迭代次数变化曲线')
plt.show()

# 用正则化优化theta
from normalEquation import normalEqn
final_theta2 = normalEqn(X2,y2)
print(computeCost(X2,y2,theta2))
