import  numpy as np
import pandas as pd
import scipy.stats as ss
# SVR回归器
from sklearn.svm import SVR
# 决策树回归器
from sklearn.tree import DecisionTreeRegressor

# 特征选择的方法
# selectKBest过滤思想
# RFE 包裹思想
# SelectFromModel 嵌入思想
from  sklearn.feature_selection import SelectKBest,RFE,SelectFromModel

df = pd.DataFrame({"A":ss.norm.rvs(size=10),"B":ss.norm.rvs(size=10),"C":ss.norm.rvs(size=10),"D":np.random.randint(low=0,high=2,size=10)})
print(df,'\n')
X = df.loc[:,["A","B","C"]]
Y = df.loc[:,"D"]

# k表示特征
skb = SelectKBest(k=2)
print(skb.fit(X,Y))
print(skb.transform(X),'\n')

# n_features_to_select 最终选择几个特征
# 选择特征子集，如正确率最高的特征子集
# 筛选模型，根据系数去掉若的特征，重复知道下降较快
# step 每迭代一次去掉多少特征
rfe = RFE(estimator=SVR(kernel="linear"),n_features_to_select=2,step=1)
print(rfe.fit_transform(X,Y),'\n')

# threshold 表示重要性因子低于多少会被去掉
sfm = SelectFromModel(estimator=DecisionTreeRegressor(),threshold=0.1)
print(sfm.fit_transform(X,Y))