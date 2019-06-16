import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles,make_blobs,make_moons
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering

n_samples = 1000
# factor 小圆大圆之间的间距
circles = make_circles(n_samples=n_samples,factor=0.5,noise=0.05)
# center_box 调整尺度，xy轴范围，cluster_std标准差，random_state保证每次位置是固定的
blobs = make_blobs(n_samples=n_samples,random_state=8,center_box=(-1,1),cluster_std=0.1)
moons = make_moons(n_samples=n_samples,noise=0.05)
random_data = np.random.rand(n_samples,2),None
colors = "bgrcmyk"
data = [circles,moons,blobs,random_data]
# n_clusters 类目,
# eps e临阈半径，min_samples最小样本点数。，密度可到达，密度相连
models = [("None",None),("Kmeans",KMeans(n_clusters=3)),("DBSCAN",DBSCAN(min_samples=3,eps=0.2)),
          ("AgglomerativeClustering",AgglomerativeClustering(n_clusters=3,linkage="ward"))]
# 轮廓系数
from sklearn.metrics import silhouette_score
f = plt.figure()
for inx,clt in enumerate(models):
    clt_name ,clt_entity = clt
    for i,dataset in enumerate(data):
        X,Y = dataset
        if not clt_entity:
            # 没有实体，label为0
            clt_res = [0 for item in range(len(X))]
        else:
            clt_entity.fit(X)
            clt_res = clt_entity.labels_.astype(np.int)
        # 行数，列数，图在行的位置
        plt.title(clt_name)
        try:
            print(clt_name,i,silhouette_score(X,clt_res))
        except:
            pass
        f.add_subplot(len(models),len(data),inx*len(data)+i+1)
        [plt.scatter(X[p,0],X[p,1],color=colors[clt_res[p]]) for p in range(len(X))]
plt.show()
