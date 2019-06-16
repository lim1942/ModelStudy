import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("data/HR.csv.bak2")

# 清洗
df = df.dropna(how="any",axis=0)
df = df[df["last_evaluation"]<=1][df["salary"]!="nme"][df["department"]!="sale"]

# 标签
# lbs = df["department"].value_counts().index
lbs = df["salary"].value_counts().index
# explodes = [0.1 if i=="sales" else 0 for i in lbs]
explodes = [0.1 if i=="low" else 0 for i in lbs]
print(explodes)
# 设置标签，显示百分比，颜色
# plt.pie(df["department"].value_counts(normalize=True),labels=lbs,autopct="%1.1f%%",colors=sns.color_palette("Reds"),explode=explodes)
plt.pie(df["salary"].value_counts(normalize=True),labels=lbs,autopct="%1.1f%%",colors=sns.color_palette("Reds"),explode=explodes)
plt.show()