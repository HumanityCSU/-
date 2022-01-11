import numpy as np

# 计算损失函数
def computeCost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))

# 梯度下降函数
def gradientDescent(X,y,theta,alpha,iters): # inters:迭代次数
    temp = np.matrix(np.zeros(theta.shape))# 构建0值矩阵
    parameters = int(theta.ravel().shape[1])# ravel计算需要求解的个数 功能是将多维数组展平成1维
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T)-y # 这里没有设置batch，直接所有的数做矩阵乘法，有可能会超过显存（内存）
        for j in range(parameters):
            term = np.multiply(error,X[:,j])# 计算两个矩阵（h^θ(x)-y）*x
            temp[0,j] = theta[0,j]-((alpha/len(X))*np.sum(term))# 参数更新公式：θ(J) = θ(J)-α(∂(J)/∂(θ(j))) , ∂:偏微分符号,α:学习率
       
        theta = temp
        cost[i] = computeCost(X,y,theta)
    return theta,cost

def batch_gradientDescent(X,y,theta,alpha,iters,batch):
    #定义每一批梯度下降样本的开始和结束
    start = 0
    end = (start+ batch)%X.shape[1]

    temp = np.matrix(np.zeros(theta.shape))# 构建0值矩阵
    parameters = int(theta.ravel().shape[1])# ravel计算需要求解的个数 功能是将多维数组展平成1维
    cost = np.zeros(iters)

    if start<X.shape[1]:
        batch_X = X[start:end-1]
        batch_y = y[start:end-1]
        for i in range(iters):
            error = (batch_X*theta.T)-batch_y 
            for j in range(parameters):
                term = np.multiply(error,batch_X[:,j])# 计算两个矩阵（h^θ(x)-y）*x
                temp[0,j] = theta[0,j]-((alpha/len(batch_X))*np.sum(term))# 参数更新公式：θ(J) = θ(J)-α(∂(J)/∂(θ(j))) , ∂:偏微分符号,α:学习率

            theta = temp
            cost[i] = computeCost(batch_X,batch_y,theta)
        
        start += batch
        end = (start+batch)%X.shape[1]

    return theta,cost