import pandas as pd


# 字符型的空值会表示为none，数字型的空值为表示为nan
df = pd.DataFrame({"A": ["a0", "a1", "a1", "a2", "a3", "a4"], "B": ["b0", "b1", "b2", "b2", "b3", None],
                   "C": [1, 2, None, 3, 4, 5], "D": [0.1, 10.2, 11.4, 8.9, 9.1, 12], "E": [10, 19, 32, 25, 8, None],
                   "F": ["f0", "f1", "g2", "f3", "f4", "f5"]})

print(df.isnull())
# 删除用控制的行
print(df.dropna())
# 只对b属性，用空值才删除一行数据
print(df.dropna(subset=["B"]))
# 针对某个属性，来删除重复行，keep选择保留哪一行，inplace参数表示是否在原df上修改
print(df.drop_duplicates(subset=["A"],keep="first"))
# 填充某个数据
print(df.fillna("b*"))
print(df.fillna(df["E"].mean()))
# 插值的方法
print(df["E"].interpolate())

# 三次样调的插值方法
print("# 三次样调的插值方法")
s2 = pd.Series([1,None,4,5,20]).interpolate(method="spline",order=3)
print(s2)

#四分位数清洗
print("#四分位数清洗")
upper_q = df["D"].quantile(0.75)
lower_q = df["D"].quantile(0.25)
q_int = upper_q -lower_q
k = 1.5
print(df[df["D"]>lower_q-k*q_int][df["D"]<upper_q+k*q_int])

#删除以g开头的异常值
print("#删除以g开头的异常值")
print(df[[True if item.startswith("f") else False for item in list(df["F"].values)]])