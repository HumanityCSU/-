from tkinter.ttk import Style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from sklearn.metrics import classification_report

data = pd.read_csv(r'./data_sets/ex2data1.txt',names = ['exam1','exam2','admitted'])
#print(data.head())

# sns.set(context = 'notebook',style='darkgrid',palette=sns.color_palette('RdBu',2))#设置样式参数，默认主题darkgrid(灰色背景+白色网格)

# sns.lmplot(x='exam1',y='exam2',hue='admitted',data = data, #hue:将name所指定的不同类型的数据叠加在一张图中显示
#             height=6, #旧版本参数是size,改成了height
#             fit_reg=False, #控制是否显示拟合的直线
#             scatter_kws={'s':50}
#           )
# plt.show()


def get_X(df) : #读取特征 
    ones = pd.DataFrame({'ones':np.ones(len(df))}) #ones是m行1列的dataframe，len(dataframe)会返回dataframe有多少行
    data = pd.concat([ones,df],axis=1) #合并数据，根据列合并axis=1的时候，concat就是行对齐，然后将不同的两张表合并 
    return np.array(data.iloc[:,:-1]) #这个操作返回ndarry,不是矩阵;data.iloc[:,:-1]会返回除了最后一列的前面所有列，正好是X_train

def get_y(df) : #读取标签
    return np.array(df.iloc[:,-1]) #df.iloc[:,-1]是指df的最后一列

def normalize_feature(df): 
    return df.apply(lambda column:(column - column.mean())/ column.std()) #lambda表达式：对后面所有的column做了一个循环(归一化)

#得到dataframe形式的X和y
X = get_X(data)
# print(X.shape)
y = get_y(data)
# print(y.shape)

def sigmoid(z) : #sigmoid函数原型：g(z) = 1 / (1+e^(-z))
    return 1 / (1+np.exp(-z))

# #展示sigmoid函数
# fig,ax = plt.subplots(figsize=(8,6))
# ax.plot(np.arange(-10,10,step=0.01),sigmoid(np.arange(-10,10,step=0.01)))
# ax.set_xlabel('z',fontsize=18)
# ax.set_ylabel('g(z)',fontsize=18)
# ax.set_title('sigmoid function',fontsize=18)
# plt.show()

def cost(theta,X,y): #损失函数定义为极大似然估计函数
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1-y) * np.log(1-sigmoid(X @ theta))) #X@theta 和X.dot(theta)等价

#初始化theta
theta = np.array([0,0,0])
# print(cost(theta,X,y))


def gradient(theta,X,y):
    return (1 / len(X) * X.T @ (sigmoid(X @ theta) - y)) #只要矩阵和向量相乘的时候shape不矛盾就行
# print(gradient(theta,X,y))

#拟合参数,用scipy.optimize.minimize:这个opt里面有一个minimize直接可以求出最小值。fun=损失函数，X0是参数向量，args传训练数据和目标数据，方法是牛顿梯度下降，jac传梯度矩阵
res = opt.minimize(fun=cost,x0=theta,args=(X,y),method='Newton-CG',jac=gradient)
# print(res)

def predict(x,theta): #预测函数
    prob = sigmoid(x @ theta)
    return (prob>0.5).astype(int) #这样变量转换也行

# #评价
# final_theta = res.x
# y_pred = predict(X,final_theta)
# print(classification_report(y,y_pred))

coef = -(res.x/res.x[2]) #由于hθ(x)>0代表y=1,反之代表y=0，所以hθ(x)=0是分界线，即为θ^T*X=0为分界线，解得x3=-(θ1 / θ3 * 1 + θ2 / θ3 * x2)

#画图
x = np.arange(130,step=0.1)
y = coef[0] + coef[1]*x
sns.set(context='notebook',style='ticks',font_scale=1.5) #默认使用notebook上下文 主题 context可以设置输出图片的尺寸(scale)
sns.lmplot(x='exam1',y='exam2',hue='admitted',data=data,
            height=6,
            fit_reg=False,
            scatter_kws={'s':25}
            )
plt.plot(x,y,'grey')
plt.xlim(0,130)
plt.ylim(0,130)
plt.title('Decision Boundary')
# plt.savefig('./逻辑回归')
plt.show()
