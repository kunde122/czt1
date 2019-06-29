import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
# iris = load_iris()
# SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)

from sklearn.feature_selection import SelectKBest
from minepy import MINE
from data_clean import DataClean

#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)



#选择K个最好的特征，返回特征选择后的数据
# SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)



for col in data.columns:
    print(col,sum(data[col].isnull()))



# data.to_csv("E:/code/addr_datat/search_159.csv")
data_xx=data.values[:,:-1]
data_yy=data.values[:,-1]
# res=SelectKBest(chi2, k=8).fit_transform(data_xx, data_yy)
SelectKBest(lambda X, Y: np.array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(data_xx, data_yy)

nums=sum(data["adrg"].isnull())



gj=0
