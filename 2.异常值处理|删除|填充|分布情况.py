import pandas as pd
import numpy as np

df = pd.read_csv("data/HR.csv.bak2")



# ===== 异常值分析
sl_s = df["satisfaction_level"]
# 判断是否有异常值
print(sl_s.isnull())
# 找出异常值
print(sl_s[sl_s.isnull()])
# 得到异常值所在的行
print(df[df["satisfaction_level"].isnull()])
# 删除有空的数据
print(sl_s.dropna())
# 填充有空的数据
sl_s.fillna(0)
# 通过四分位数得到上下界，分析连续异常值
# 第一四分位数 (Q1)，又称“较小四分位数”，等于该样本中所有数值由小到大排列后第25%的数字。
# 第二四分位数 (Q2)，又称“中位数”，等于该样本中所有数值由小到大排列后第50%的数字。
# 第三四分位数 (Q3)，又称“较大四分位数”，等于该样本中所有数值由小到大排列后第75%的数字。
# 第三四分位数与第一四分位数的差距又称四分位距（InterQuartile Range,IQR）。
print(sl_s.dropna().quantile(0.25))
print(sl_s.dropna().quantile(0.75))
# 负偏，大多的数据大于平均值
print(sl_s.dropna().skew())
# 负值，较正态分布较为平缓
print(sl_s.dropna().kurt())

# 利用np获取数据的分布情况，在0.0到0.1有个195个数据
print(np.histogram(sl_s.dropna().values,bins=np.arange(0.0,1.1,0.1)))



