import numpy as np
from sklearn.preprocessing import Normalizer


# l1 l2 正规化，将向量的长度正规到长度1
# 直接用在特征上
# 用在一个对象上，某个特征的影响大小，用的最多
# 模型的参数上，什么参数对模型影响大（回归模型使用最多）
print(Normalizer(norm="l1").fit_transform(np.array([[1,1,3,-1,2]])))
print(Normalizer(norm="l2").fit_transform(np.array([[1,1,3,-1,2]])))