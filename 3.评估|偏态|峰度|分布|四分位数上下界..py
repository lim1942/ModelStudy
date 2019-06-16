import pandas as pd
import numpy as np

df = pd.read_csv("data/HR.csv.bak2")



le_s = df["last_evaluation"]
print(le_s[le_s.isnull()])

print(le_s.mean())


np_s = df["number_project"]
print(np_s[np_s.isnull()])
print(np_s.mean())
print(np_s.median())
print(np_s.max())
print(np_s.min())
# 正偏
print(np_s.skew())
# 比正态分布缓和
print(np_s.kurt())
# 比例的分布
print(np_s.value_counts(normalize=True).sort_index())


amh_s = df["average_monthly_hours"]
print(amh_s.mean())
print(amh_s.std())
print(amh_s.max())
print(amh_s.min())
print(amh_s.skew())
# 平缓，聚拢程度小
print(amh_s.kurt())

# 去除上下界之外的数据
amh_s = amh_s[amh_s<amh_s.quantile(0.75)+1.5*(amh_s.quantile(0.75)-amh_s.quantile(0.25))][amh_s>amh_s.quantile(0.25)-1.5*(amh_s.quantile(0.75)-amh_s.quantile(0.25))]
print(np.histogram(amh_s.values,bins=10))
# 左闭右开
print(np.histogram(amh_s.values,bins=np.arange(amh_s.min(),amh_s.max()+10,10)))
# 左开右闭
print(amh_s.value_counts(bins=np.arange(amh_s.min(),amh_s.max()+10,10)))



tsc_s = df["time_spend_company"]
print(tsc_s.value_counts().sort_index())
print(tsc_s.value_counts().sort_index())


wa_s = df["Work_accident"]
print(wa_s.value_counts())
# 事故率
print(wa_s.mean())


l_s = df["left"]
print(l_s.value_counts())


pl5_s = df["promotion_last_5years"]
print(pl5_s.value_counts())


s_s = df['salary']
print(s_s.value_counts())

print(s_s.where(s_s!="nme").dropna())


d_s = df['department']
print(d_s.value_counts(normalize=True))