import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("data/HR.csv.bak2")

# 清洗
df = df.dropna(how="any",axis=0)
df = df[df["last_evaluation"]<=1][df["salary"]!="nme"][df["department"]!="sale"]

# 设置为横向的箱线图，saturation设置四位数边界时，whis为k值
sns.boxplot(x=df["time_spend_company"],saturation=0.75,whis=3)
plt.show()