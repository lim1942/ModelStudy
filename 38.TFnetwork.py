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
    # 决策树
    from sklearn.tree import DecisionTreeClassifier
    # 支持向量机
    from sklearn.svm import SVC
    # 随机森林
    from sklearn.ensemble import RandomForestClassifier
    # Adaboost
    from sklearn.ensemble import AdaBoostClassifier
    # 逻辑斯特
    from sklearn.linear_model import LogisticRegression


    # Sequential 人工神经网络的容器
    from keras.models import Sequential
    # Dense 神经网络层，稠密层，Activation 激活函数
    from keras.layers.core import Dense, Activation
    # SGD 随机梯度下降算法
    from keras.optimizers import SGD
    mdl = Sequential()
    mdl.add(Dense(50,input_dim=len(f_v[0])))
    mdl.add(Activation("sigmoid"))
    mdl.add(Dense(2))
    mdl.add(Activation("softmax"))
    # 优化器，lr学习率相当于梯度下降的α
    # 随机梯度下降算法
    sgd = SGD(lr=0.06)
    # 随机梯度下降算法的变种
    # 缓存adam优化器减少lose
    mdl.compile(loss="mean_squared_error",optimizer="adam")
    # nb_epoch 训练的批次,batch_size每次使用的条数
    mdl.fit(X_train,np.array([[0,1] if i==1 else [1,0] for i in Y_train]),nb_epoch=10000,batch_size=8999)
    xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]

    for i in range(len(xy_lst)):
        X_part = xy_lst[i][0]
        Y_part = xy_lst[i][1]
        # predict_classes 方法输出离散值
        Y_pred =mdl.predict_classes(X_part)
        print(i)
        print("NN", "-ACC", accuracy_score(Y_part, Y_pred))
        print("NN", "-REC", recall_score(Y_part, Y_pred))
        print("NN", "-Fl", f1_score(Y_part, Y_pred))

    return

    models = []
    models.append(("KNN",KNeighborsClassifier(n_neighbors=3)))
    models.append(("GaussianNB",GaussianNB()))
    models.append(("BernoulliNB",BernoulliNB()))
    models.append(("DecisionTreeClassifier",DecisionTreeClassifier()))
    # 惩罚度，防止错分
    models.append(("SVM Classifier",SVC(C=100)))
    # 树的个数，bootstrap：true是无放回
    models.append(("RandomForestClassifier",RandomForestClassifier(n_estimators=81,max_features=None,bootstrap=True)))
    models.append(("Adaboost",AdaBoostClassifier(base_estimator=SVC(),n_estimators=100,algorithm="SAMME")))
    models.append(("LogisticRegression",LogisticRegression(C=1000,tol=1e-10)))

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

def regr_test(features,label):
    print("X",features)
    print("Y",label)
    from sklearn.linear_model import LinearRegression,Ridge,Lasso
    #regr=LinearRegression()
    regr=Ridge(alpha=1)
    regr.fit(features.values,label.values)
    Y_pred=regr.predict(features.values)
    # 拟合出来确定的参数
    print("Coef:",regr.coef_)
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    # 平均平方误差
    print("MSE:",mean_squared_error(label.values,Y_pred))
    print("MAE:",mean_absolute_error(label.values,Y_pred))
    print("R2:",r2_score(label.values,Y_pred))

def main():
    features,label = hr_preprocessing(slr=True)
    # regr_test(features[["number_project","average_monthly_hours"]],features["last_evaluation"])

    hr_modeling(features,label)


main()

