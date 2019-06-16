import pandas as pd
import numpy as np
# 设置打印展示输出
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)


"""对比分析"""

df = pd.read_csv("data/HR.csv.bak2")

# drop行，只要出现就删行
df = df.dropna(axis=0,how="any")
# 清洗
df = df[df["last_evaluation"]<=1][df["salary"]!="nme"][df["department"]!="sale"]

# 分组分析
print(df.groupby("department").mean())

# 取两个列对比分析
# print(df.loc[:,["last_evaluation","department"]].groupby("department").mean())
print(df.loc[:,["average_monthly_hours","department"]].groupby("department")["average_monthly_hours"].apply(lambda x:x.max()-x.min()))

