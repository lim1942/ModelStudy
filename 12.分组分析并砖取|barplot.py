import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

# 分组分析
sns.set_context(font_scale=1.5)
df = pd.read_csv("data/HR.csv.bak2")
# 向下砖取选择department
sns.barplot(x="salary",y="left",hue="department",data=df)
sl_s = df["satisfaction_level"]
# sns.barplot(list(range(len(sl_s))),sl_s.sort_values())
plt.show()
