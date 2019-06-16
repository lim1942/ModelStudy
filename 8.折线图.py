import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("data/HR.csv.bak2")

# 清洗
df = df.dropna(how="any",axis=0)
df = df[df["last_evaluation"]<=1][df["salary"]!="nme"][df["department"]!="sale"]
# 折线图
# 分组后对各组所有取均值，针对所有列
sub_df = df.groupby("time_spend_company").mean()
# sns.pointplot(sub_df.index,sub_df["left"])
# 另一种画图法
sns.pointplot(x="time_spend_company",y="left",data=df)
plt.show()