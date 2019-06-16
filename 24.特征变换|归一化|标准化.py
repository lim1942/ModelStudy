import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler


# 归一化，归到（0，1）
print(MinMaxScaler().fit_transform(np.array([1,4,10,15,21]).reshape(-1,1)))

# 标准化，放大数据与其他数据的大小关系
# 均值为0，标准差为1
print(StandardScaler().fit_transform(np.array([1,1,1,1,0,0,0,0]).reshape(-1,1)))