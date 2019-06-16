import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

# 交叉分析

df = pd.read_csv("data/HR.csv.bak2")
# 得到分组后的各分组的索引
dp_indices = df.groupby(by="department").indices
sales_values = df["left"].iloc[dp_indices["sales"]].values
techical_values = df["left"].iloc[dp_indices["technical"]].values
print(ss.ttest_ind(sales_values,techical_values))
# 两两部门间求离职的相关性
dp_keys = list(dp_indices.keys())
dp_t_mat = np.zeros([len(dp_keys),len(dp_keys)])
for i in range(len(dp_keys)):
    for j in range(len(dp_keys)):
        p_value = ss.ttest_ind(df["left"].iloc[dp_indices[dp_keys[i]]].values, \
                  df["left"].iloc[dp_indices[dp_keys[j]]].values)[1]
        # p值小于某值时置为-1
        if p_value <0.05:
            dp_t_mat[i][j]=-1
        dp_t_mat[i][j]=p_value
sns.heatmap(dp_t_mat,xticklabels=dp_keys,yticklabels=dp_keys)
plt.show()


# # 透视表，aggfunc聚合方法
# piv_tb = pd.pivot_table(df,values="left",index=["promotion_last_5years","salary"],\
#                         columns=["Work_accident"],aggfunc=np.mean)
# print(piv_tb)
# sns.heatmap(piv_tb,vmin=0,vmax=1)
# plt.show()
