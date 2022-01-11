import numpy as np

# 正规方程(一步到位算theta)
def normalEqn(X,y):
    theta = np.linalg.inv(X.T@X)@X.T@y # np.linalg.inv是算矩阵的逆，另外@是矩阵之间的乘号，在numpy里面 
    return theta
