import numpy as np
#PCA,
# sklearn 奇异值分解的方法来降维
print("\n奇异值降维")
data = np.array([np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]),
                 np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
                 ]).T
from sklearn.decomposition import PCA
lower_dim=PCA(n_components=1)
lower_dim.fit(data)
print(lower_dim.fit_transform(data))

def myPCA(data,n_components=1000000):
    # 针对列来求均值
    mean_vals = np.mean(data,axis=0)
    mid = data - mean_vals
    # false 表示列的协方差矩阵
    cov_mat = np.cov(mid,rowvar=False)
    from scipy import linalg
    # 求协方差矩阵的特征值和特征向量
    eig_vals,eig_vects = linalg.eig(np.mat(cov_mat))
    # 取出最大的特征值对应的最大的特征向量
    eig_val_index = np.argsort(eig_vals) #得到排序后的下标
    eig_val_index = eig_val_index[:-(n_components+1):-1]
    # 得到特征向量
    eig_vects = eig_vects[:,eig_val_index]
    low_dim_mat = np.dot(mid,eig_vects)
    return  low_dim_mat,eig_vals

data = np.array([np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]),
                 np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
                 ]).T

print("\n协方差降维")
print(myPCA(data,1))