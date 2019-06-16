import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

# 相关分析
# 相关性分析，找出对left影响最大的特征
sns.set_context(font_scale=1.5)
df = pd.read_csv("data/HR.csv.bak2")
sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap=sns.color_palette("RdBu",n_colors=128))
plt.show()

s1 = pd.Series(["X1","X1","X2","X2","X2","X2"])
s2 = pd.Series(["Y1","Y1","Y1","Y2","Y2","Y2"])

# ===========对于离散型变量=========

# 得到熵
def getEntropy(s):
    if not isinstance(s,pd.core.series.Series):
        s = pd.Series(s)
    # 得到分布
    prt_ary = pd.groupby(s,by=s).count().values/float(len(s))
    return -(np.log2(prt_ary)*prt_ary).sum()
print("getEntropy:",getEntropy(s2))

# 条件熵
def getCondEntropy(s1,s2):
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]]=d.get(s1[i],[]) + [s2[i]]
    return  sum([getEntropy(d[k])*len(d[k])/float(len(s1)) for k in d])
print("getCondEntropy:",getCondEntropy(s1,s2))
print("getCondEntropy:",getCondEntropy(s2,s1))

# 互信息，熵增益
def getEntropyGain(s1,s2):
    return getEntropy(s2)-getCondEntropy(s1,s2)
print("getEntropyGain:",getEntropyGain(s1,s2))

# 熵增益率,不对称
def getEntropyGainRatio(s1,s2):
    return getEntropyGain(s1,s2)/getEntropy(s2)
print("getEntropyGainRatio:",getEntropyGainRatio(s1,s2))

# 衡量离散值的相关性
import math
def getDiscreteCorr(s1,s2):
    return getEntropyGain(s1,s2)/math.sqrt(getEntropy(s1)*getEntropy(s2))
print("getDiscreteCorr:",getDiscreteCorr(s1,s2))




# ======= 基尼系数，对于连续型变量 =========
def getprobSS(s):
    if not isinstance(s,pd.core.series.Series):
        s = pd.Series(s)
    # 得到分布
    prt_ary = pd.groupby(s,by=s).count().values/float(len(s))
    return  sum(prt_ary**2)
def getGini(s1,s2):
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]]=d.get(s1[i],[]) + [s2[i]]
    return 1-sum([getprobSS(d[k])*len(d[k])/float(len(s1)) for k in d])
print("Gini:",getGini(s1,s2))
print("Gini:",getGini(s2,s1))
