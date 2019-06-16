import pandas as pd
import scipy.stats as ss

df = pd.read_csv("data/HR.csv.bak2")


# # 均值
# print(df["satisfaction_level"].mean())
# # 中位数
# print(df["satisfaction_level"].median())
# # 四分位数
# print(df["satisfaction_level"].quantile(q=0.25))
# # 众数，返回一个series，因为可能有多个
# print(df["satisfaction_level"].mode())
#
# # 标注差
# print(df["satisfaction_level"].std())
# # 方差
# print(df["satisfaction_level"].var())
# # 求和
# print(df["satisfaction_level"].sum())
#
#
# # 偏态系数，负偏，大多人比较满意
# print(df["satisfaction_level"].skew())
# # 峰态系数
# print(df["satisfaction_level"].kurt())



# # 标准正态分布对象
# # 均值0，方差1，偏态系数0，偏态系数0
# print(ss.norm.stats(moments="mvsk"))
# # 分布函数，指定横坐标，返回纵坐标的值，参数是横坐标
# print(ss.norm.pdf(0.0))
# # 返回从负无穷，积分到0.9时的横坐标。参数是概率的积分
# print(ss.norm.ppf(0.9))
# # 从负无穷积分到某一横坐标时，概率的积分值。参数是横坐标（标注差的倍数）
# print(ss.norm.cdf(2))
# # 分布在正负两个标准差之间的概率
# print(ss.norm.cdf(2)-ss.norm.cdf(-2))
# # 得到十个符合正态分布的数字
# print(ss.norm.rvs(size=10))


# # 卡方分布
# ss.chi2
# # t分布
# ss.t
# # f 分布
# ss.f


# 抽样10个
print(df.sample(10))
# 抽0.01的样本
print(df.sample(frac=0.01))
