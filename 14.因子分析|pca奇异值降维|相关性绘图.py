import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns



# 设置打印展示输出
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)


# 因子分析
sns.set_context(font_scale=1.5)
df = pd.read_csv("data/HR.csv.bak2")
# 主成分分析工具
from sklearn.decomposition import PCA
my_pca = PCA(n_components=4)
# 不能出现离散的属性,和要分析的属性，。axis指定为列
lower_mat = my_pca.fit_transform(df.drop(labels=["salary","department","left"],axis=1))
print("Ratio:",my_pca.explained_variance_ratio_)
print(lower_mat)
sns.heatmap(pd.DataFrame(lower_mat).corr(),vmin=-1,vmax=1,cmap=sns.color_palette("RdBu",n_colors=128))
plt.show()