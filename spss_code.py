import os
import pandas as pd
# RECODE NL (Lowest thru 40=1) (40 thru 60=2) (60 thru Highest=3) INTO analysis_年龄.
# VARIABLE LABELS  analysis_年龄 '年龄分段'.
# EXECUTE.

def anova_test_find(in_path,out_path):
    data = pd.read_excel(in_path)
    data.dropna(axis=0, how='any',inplace=True)
    # data.to_excel("E:/code/addr_datat/search_159_10w_proc22.xlsx")
    print("length of row:{}".format(len(data)))
    # var="adrg+NL+SJZYTS+LYFS+JBDM+createTime+QKDJ1+QKYHLB1+JBCOUNT+SSCOUNT"
    xx_var="C(adrg)+C(NL)+C(SJZYTS)+C(LYFS)+C(JBDM)+C(createTime)+C(QKDJ1)+C(QKYHLB1)+C(JBCOUNT)+C(SSCOUNT)"
    yy_var="yy_data"

    from statsmodels.formula.api import ols
    model=ols("{}~{}".format(yy_var,xx_var),data).fit()
    parameter = pd.DataFrame(model.tvalues)
    pvalues_param=pd.DataFrame(model.pvalues)
    indexs=pvalues_param.index
    max_tvalue={}
    for ind,value in zip(pvalues_param.index,pvalues_param.values):
        if abs(value[0])<0.05:
            max_tvalue[ind]=value[0]
            print(ind,value)
    # for ind in indexs:
    #     if pvalues_param[ind]<0.05:
    #         max_tvalue[ind]=pvalues_param[ind,0]




    # columns=parameter.columns
    # from statsmodels.stats.outliers_influence import summary_table
    # st, data, ss2 = summary_table(model, alpha=0.05)

    max_tvalue=sorted(max_tvalue.items(),key=lambda x:x[1],reverse=True)
    for key in max_tvalue:
        print(key,max_tvalue[key])

    # print(model.summary())
    # print(max_tvalue)




data_path="E:/code/addr_datat/search_159_10w_proc11.xlsx"
path="./tmp/out.txt"
str=['adrg']

import pandas as pd
# data=pd.read_excel(data_path)


def get_sub_names(dataframe,col):
    names = set()
    for item in dataframe[col]:
        if not pd.isnull(item):
            names.add(item)
    return names

def create_dumy(col_names,out_path):
    with open(path,'w')  as fw:
        for col_n in col_names:
            sub_names=get_sub_names(data,col_n)
            for name in sub_names:
                fw.write("DO IF({}='{}').".format(col_n,name)+'\n')
                fw.write("compute {}_{}=1.".format(col_n, name) + '\n')
                fw.write('ELSE.\n')
                fw.write("compute {}_{}=0.".format(col_n, name) + '\n')
                fw.write('END IF.\n')
                fw.write('EXECUTE.\n')
                fw.write('\n')
            fw.write('\n')

def create_dumy2(col_name,sub_names,out_path):
    with open(out_path,'w')  as fw:
            for name in sub_names:
                fw.write("DO IF({}='{}').".format(col_name,name)+'\n')
                fw.write("compute {}_{}=1.".format(col_name, name) + '\n')
                fw.write('ELSE.\n')
                fw.write("compute {}_{}=0.".format(col_name, name) + '\n')
                fw.write('END IF.\n')
                fw.write('EXECUTE.\n')
                fw.write('\n')
            fw.write('\n')
# exit(0)


