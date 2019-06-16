import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 10000)

# s1: satisfaction_level  -- False:MinMaxScalr; True:StandardScaler
# le: last_evaluation  -- False:MinMaxScalr; True:StandardScaler
# npr: number_project  -- False:MinMaxScalr; True:StandardScaler
# amh: average_monthly_hours  -- False:MinMaxScalr; True:StandardScaler
# tsc: time_spend_company  -- False:MinMaxScalr; True:StandardScaler
# wa: Work_accident  -- False:MinMaxScalr; True:StandardScaler
# pl5: promotion_last_5years  -- False:MinMaxScalr; True:StandardScaler
# dp:department  -- False:labelEncoding ; True: OneHotEncoding
# slr:salary  -- False:labelEncoding ; True: OneHotEncoding
# lower_d 是否降维
# ld_n :降维后的特征个数
def hr_preprocessing(sl=False,le=False,npr=False,amh=False,tsc=False,wa=False,pl5=False,slr=False,dp=False,lower_d=False,ld_n=1):
    df = pd.read_csv("./data/HR.csv.bak2")
    # 1 得到标注
    label = df["left"]
    df = df.drop("left",axis=1)
    # 2 清洗数据
    df = df.dropna(subset=["satisfaction_level","last_evaluation"])
    df = df[df["satisfaction_level"]<=1][df["salary"]!="nme"]
    # 3 特征选择
    # 4 特征处理  (归一化和标准化）
    scaler_lst = [sl,le,npr,amh,tsc,wa,pl5]
    column_lst = ["satisfaction_level","last_evaluation","number_project",
                  "average_monthly_hours","time_spend_company","Work_accident",
                  "promotion_last_5years"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            df[column_lst[i]]=\
            MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]
        else:
            df[column_lst[i]] = \
            StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]
    # 5 特征处理，数值化,labernecode,onehot
    scaler_lst = [slr,dp]
    column_lst = ["salary","department"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i] == "salary":
                df[column_lst[i]] = [map_salary(s) for s in df["salary"].values]
            else:
                df[column_lst[i]] = LabelEncoder().fit_transform((df[column_lst[i]]))
            df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]
        else:
            df = pd.get_dummies(df,columns=[column_lst[i]])
    # 6 降维
    if lower_d:
        # 因为这里标注（left）的类个数为2，降维后的维度只能小于等于1
        # return LinearDiscriminantAnalysis(n_components=ld_n)
        return PCA(n_components=ld_n).fit_transform(df.values),label
    return  df,label


def hr_modeling(features,label):
    # 切分训练集和测试集
    from sklearn.model_selection import train_test_split
    f_v = features.values
    l_v = label
    # 得到验证集
    X_tt,X_validation,Y_tt,Y_validation = train_test_split(f_v,l_v,test_size=0.2)
    # 得到训练集和测试集
    X_train,X_test,Y_train,Y_test = train_test_split(X_tt,Y_tt,test_size=0.25)

    # 准确率，召回率，f值
    from sklearn.metrics import  accuracy_score,recall_score,f1_score
    #KNN ,连续值好
    from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
    # 引入朴素贝叶斯，高斯和伯努利。在离散下表现更好
    from sklearn.naive_bayes import GaussianNB,BernoulliNB
    from sklearn.tree import DecisionTreeClassifier

    models = []
    models.append(("KNN",KNeighborsClassifier(n_neighbors=3)))
    models.append(("GaussianNB",GaussianNB()))
    models.append(("BernoulliNB",BernoulliNB()))
    models.append(("DecisionTreeClassifier",DecisionTreeClassifier()))
    for cls_name,clf in models:
        clf.fit(X_train,Y_train)
        xy_lst = [(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(i)
            print(cls_name,"-ACC",accuracy_score(Y_part,Y_pred))
            print(cls_name,"-REC",recall_score(Y_part,Y_pred))
            print(cls_name,"-Fl",f1_score(Y_part,Y_pred))


# 映射到数字
d = dict([("low",0),("medium",1),("high",2)])
def map_salary(s):
    return d.get(s,0)


def main():
    features,label = hr_preprocessing(slr=True)
    hr_modeling(features,label)


main()