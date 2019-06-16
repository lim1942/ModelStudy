import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 标签化
print(LabelEncoder().fit_transform(np.array(["Down","up","Down"])))


# oneHot,独热编码
lb_encoder = LabelEncoder()
lb_tran_f = lb_encoder.fit_transform(np.array(["Red","Yellow","Blue","Green"]))
# OneHotEncoder 需要标签化后的对象，并reshape
# 必须reshape
oht_encoder = OneHotEncoder().fit(lb_tran_f.reshape(-1,1))
# 得到稀疏矩阵
print(oht_encoder.transform(lb_encoder.transform(np.array(["Yellow","Blue","Green","Green","Red"])).reshape(-1,1)).toarray())
