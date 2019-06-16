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



# 映射到数字
d = dict([("low",0),("medium",1),("high",2)])
def map_salary(s):
    return d.get(s,0)

# 线性回归
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
    regr_test(features[["number_project","average_monthly_hours"]],features["last_evaluation"])



main()





# 梯度
# import numpy as np
#
# # Size of the points dataset.
# m = 20
#
# # Points x-coordinate and dummy value (x0, x1).
# X0 = np.ones((m, 1))
# X1 = np.arange(1, m+1).reshape(m, 1)
# X = np.hstack((X0, X1))
#
# # Points y-coordinate
# y = np.array([
#     3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
#     11, 13, 13, 16, 17, 18, 17, 19, 21
# ]).reshape(m, 1)
#
# # The Learning Rate alpha.
# alpha = 0.01
#
# def error_function(theta, X, y):
#     '''Error function J definition.'''
#     diff = np.dot(X, theta) - y
#     return (1./2*m) * np.dot(np.transpose(diff), diff)
#
# def gradient_function(theta, X, y):
#     '''Gradient of the function J definition.'''
#     diff = np.dot(X, theta) - y
#     return (1./m) * np.dot(np.transpose(X), diff)
#
# def gradient_descent(X, y, alpha):
#     '''Perform gradient descent.'''
#     theta = np.array([1, 1]).reshape(2, 1)
#     gradient = gradient_function(theta, X, y)
#     while not np.all(np.absolute(gradient) <= 1e-5):
#         theta = theta - alpha * gradient
#         gradient = gradient_function(theta, X, y)
#     return theta
#
# optimal = gradient_descent(X, y, alpha)
# print('optimal:', optimal)
# print('error function:', error_function(optimal, X, y)[0,0])