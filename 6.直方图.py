import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("data/HR.csv.bak2")

# 清洗
df = df.dropna(how="any",axis=0)
df = df[df["last_evaluation"]<=1][df["salary"]!="nme"][df["department"]!="sale"]

# sns.set_style(style="darkgrid")
sns.set_style(style="whitegrid")
# 设置字体样式和大小
sns.set_context(context="poster",font_scale=0.8)
# 设置方条的颜色
sns.set_palette(sns.color_palette("RdBu",n_colors=7))
# 直接用seaborn画图

f = plt.figure()
f.add_subplot(1,3,1)
# kde ，hist分别控制分布图和直方图是否展示
sns.distplot(df["satisfaction_level"],bins=10,kde=True,hist=True)
f.add_subplot(1,3,2)
sns.distplot(df["last_evaluation"],bins=10)
f.add_subplot(1,3,3)
sns.distplot(df["average_monthly_hours"],bins=10 )
plt.show()