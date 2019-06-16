import numpy as np
import pandas as pd
import scipy.stats as ss

# 获得正态分布并检验、
print("正态分布检验--")
norm_dist = ss.norm.rvs(size=1000)
print(ss.normaltest(norm_dist))

# 检验卡方分布，p值越小越显著
print("\n卡方检验")
print(ss.chi2_contingency([[15,95],[85,5]]))
# 检验统计量，p值，自由度，理论分布
# NormaltestResult(statistic=1.3921064423225058, pvalue=0.49854908892368155)
# (126.08080808080808, 2.9521414005078985e-29, 1, array([[55., 55.],
#        [45., 45.]]))

# 独立t分布检验，
print("\nt检验")
print(ss.ttest_ind(ss.norm.rvs(size=10000),ss.norm.rvs(size=10000)))

# 方差检验检验两组数据的均值有没有大的差异
print("\n方差检验")
print(ss.f_oneway([49,50,39,40,43],[28,31,30,26,34],[38,40,45,42,48]))

# # qq图检验是不是正态分布
from statsmodels.graphics.api import  qqplot
from matplotlib import pyplot as plt
qqplot(ss.norm.rvs(size=100))
plt.show()


# 相关系数，皮尔逊，斯皮尔曼
print("\n相关系数")
s1 = pd.Series([0.1,0.2,1.1,2.4,1.3,0.3,0.5])
s2 = pd.Series([0.5,0.4,1.2,2.5,1.1,0.7,0.1])
print("皮尔逊")
print(s1.corr(s2))
print("斯皮尔曼")
print(s1.corr(s2,method="spearman"))
# # 矩阵的相关系数，皮尔逊，斯皮尔曼
print("矩阵相关系数")
df = pd.DataFrame([s1,s2])
print(df.corr())
print("矩阵秩的相关系数")
df = pd.DataFrame(np.array([s1,s2]).T)
print(df.corr())


# 线性回归
print("\n线性回归")
x = np.arange(10).astype(np.float).reshape((10,1))
y = x*3 + 4 + np.random.random((10,1))
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
res = reg.fit(x,y)
y_pred = reg.predict((x))
print("============ x ")
print(x)
print("============ y")
print(y)
print("============ y的预测值")
print(y_pred)
print("============ 参数")
print(reg.coef_)
print("============ 截距")
print(reg.intercept_)


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
