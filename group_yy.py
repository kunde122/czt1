import pandas as pd

path="E:/code/addr_datat/excel_dum.xlsx"

data=pd.read_excel(path)
import numpy as np
res=np.array(data["yy_data"])
split=pd.qcut(res,20)
labels=split.labels
data["yy_group"]=labels
data.to_excel("E:/code/addr_datat/excel_dum11.xlsx")