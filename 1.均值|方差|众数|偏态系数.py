import pandas as pd
import scipy.stats as ss

df = pd.read_csv("data/HR.csv.bak2")


# 均值
print(df["satisfaction_level"].mean())
# 中位数
print(df["satisfaction_level"].median())
# 四分位数
print(df["satisfaction_level"].quantile(q=0.25))
# 众数，返回一个series，因为可能有多个
print(df["satisfaction_level"].mode())

# 标注差  衡量离散程度，和数据在同一个量纲
print(df["satisfaction_level"].std())
# 方差  衡量离散程度
print(df["satisfaction_level"].var())
# 求和
print(df["satisfaction_level"].sum())


# 正态分布的对比量
# 偏态系数，负偏，大多的数据大于平均值
# 偏度的衡量是相对于正态分布来说，正态分布的偏度为0。因此我们说，若数据分布是对称的，偏度为0.若偏度>0,则可认为分布为右偏，即分布有一条长尾在右；若偏度<0，则可认为分布为左偏
print(df["satisfaction_level"].skew())
# 峰态系数   峰度的取值范围为[1,+∞)，完全服从正态分布的数据的峰度值为 3，峰度值越大，概率分布图越高尖，峰度值越小，越矮胖。
print(df["satisfaction_level"].kurt())



# 标准正态分布对象(高斯分布)
# 均值0，方差1，偏态系数0，偏态系数0
print(ss.norm.stats(moments="mvsk"))
# 分布函数，指定横坐标，返回纵坐标的值，参数是横坐标
print(ss.norm.pdf(0.0))
# 返回从负无穷，积分到0.9时的横坐标。参数是概率的积分
print(ss.norm.ppf(0.9))
# 从负无穷积分到某一横坐标时，概率的积分值。参数是横坐标（标注差的倍数）
print(ss.norm.cdf(2))
# 分布在正负两个标准差之间的概率
print(ss.norm.cdf(2)-ss.norm.cdf(-2))
# 得到十个符合正态分布的数字
print(ss.norm.rvs(size=10))


# 卡方分布
ss.chi2
# t分布
ss.t
# f 分布
ss.f


# 抽样10个
print(df.sample(10))
# 抽0.01的样本
print(df.sample(frac=0.01))
