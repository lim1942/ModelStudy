import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 设置打印展示输出
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
df = pd.read_csv("data/HR.csv.bak2")

# # sns.set_style(style="darkgrid")
# sns.set_style(style="whitegrid")
# # 设置字体样式和大小
# sns.set_context(context="poster",font_scale=0.8)
# # 设置方条的颜色
# sns.set_palette("Reds")
# sns.set_palette(sns.color_palette("RdBu",n_colors=7))
# # 直接用seaborn画图 ,按照salary 分成 低-中-高。再在低中高看部门的value_counts
# sns.countplot(x="salary",data=df,hue="department")
# plt.show()

# # 设置标题
plt.title("Salary")
# x轴名称
plt.xlabel("salary")
# y轴名称
plt.ylabel("num")
# 设置x轴坐标，并平移
plt.xticks(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts().index)
# 设置x，y轴范围
plt.axis([0,4,0,10000])
plt.bar(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts(),width=0.5)
# 标注数值,ha,va水平垂直位置
for x,y in zip(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts()):
    plt.text(x,y,y,ha="center",va="bottom")

plt.show()


