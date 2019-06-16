import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 与pca，特征向量的无监督降维不同，lda是有监督的降维
# 投影变换后同一标注的距离尽可能小，不同标注间进可能大
X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
Y = np.array([1,1,1,2,2,2])

# 降到一维
print(LinearDiscriminantAnalysis(n_components=1).fit_transform(X,Y))

# fisher分类器
clf = LinearDiscriminantAnalysis(n_components=1).fit(X,Y)
print(clf.predict([[0.8,1]]))