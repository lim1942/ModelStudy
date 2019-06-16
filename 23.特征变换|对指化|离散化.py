import numpy as np
import pandas as pd


lst = [6,8,10,15,16,24,25,40,67]
# 等深分箱
print(pd.qcut(lst,q=3))
print(pd.qcut(lst,q=3,labels=["low","medium","high"]),"\n\n")

# 等宽封箱
print(pd.cut(lst,bins=3,labels=["low","medium","high"]))