import pandas as pd
import os

path="E:/code/addr_datat/excel_dum.xlsx"
path='D:\code\data'
data=pd.DataFrame()
for file in os.listdir(path):
    file_path=os.path.join(path,file)
    tmp=pd.read_csv(file_path)
    tmp.drop(['USERNAME_'],axis=1,inplace=True)
    data=pd.concat([data,tmp])
    pass
print(len(data))
data.to_csv('./tmp/origin1.csv',index=False,encoding='gbk')

# data=pd.read_excel(path)
# import numpy as np
# res=np.array(data["yy_data"])
# split=pd.qcut(res,20)
# labels=split.labels
# data["yy_group"]=labels
# data.to_excel("E:/code/addr_datat/excel_dum11.xlsx")