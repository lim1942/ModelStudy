import numpy as np
import pandas as pd
import scipy.stats as ss

# ============= 一.单变量 获得正态分布并检验、
print("正态分布检验--")
norm_dist = ss.norm.rvs(size=1000)
print(ss.normaltest(norm_dist))

# ============= 二 单变量 不同样本之间的对比分析
# 1.检验卡方分布，p值越小越显著。卡方值可用于连续型变量的卡方分箱（先分成很多组，然后把相邻切卡方值和小的两组合并一起，不断合并，即为卡方分箱）
# p=0.01就是有99%的可信度接受他们有相关性
# 卡方值，p值，自由度，理论分布
print("\n卡方检验")
print(ss.chi2_contingency([[15,95],[85,5]]))
# 2.独立t分布检验，t分布用于检验均值是否不同，只能用于连续型变量
print("\nt检验")
print(ss.ttest_ind(ss.norm.rvs(size=10000),ss.norm.rvs(size=10000)))
# 3.方差检验检验两组数据的方差有没有大的差异，只能用于连续型变量
print("\n方差检验")
print(ss.f_oneway([49,50,39,40,43],[28,31,30,26,34],[38,40,45,42,48]))


# # # 三. 单变量 qq图检验是不是正态分布
from statsmodels.graphics.api import  qqplot
from matplotlib import pyplot as plt
qqplot(ss.norm.rvs(size=100))
plt.show()


# ================== 四.两个变量 相关系数，皮尔逊，斯皮尔曼,越接近1相关性越大，基于协方差计算出来的，协方差除以(两者标准差的积)
# 系数为0时不相关，接近1正相关，-1负相关
# 通常bai情况下默认用pearson相关系数，数据分布du呈现出不正态时用Spearman相关系数。
# 1.连续数据，正态bai分布，线性关系，用皮尔逊相关du系数zhi是dao最恰当，当然用spearman相关系数也可以，效率没有pearson相关系数高。
# 2.上述任一条件不满足，就用spearman相关系数，不能用pearson相关系数。
# 3.两个定序测量数据之间也用spearman相关系数，不能用pearson相关系数。
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


# =================== 五. 两个变量 线性回归
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



