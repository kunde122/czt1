#coding:utf-8
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# myfont = matplotlib.font_manager.FontProperties(fname='E:/code/fonts/yahei.ttf')
# matplotlib.rcParams['axes.unicode_minus'] = False
# decisionNode = dict(boxstyle="sawtooth", fc="0.8")
# leafNode = dict(boxstyle="round4", fc="0.8")
# arrow_args = dict(arrowstyle="<-")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
np.random.seed(12)
# x = np.random.normal(size=100)
# sns.set()
# sns.distplot(x)
# t=np.arange(0,1,0.1)
# plt.plot([0]*len(t),t,lw=2,label='yre')
# plt.annotate('local max', xy=(0, 0), xytext=(0, 0),xycoords='data',
#              arrowprops=dict(facecolor='black', shrink=0.05))

# plt.show()


def old():
    in_path="E:/code/addr_datat/618data.xlsx"
    data = pd.read_csv("E:/code/addr_datat/618data.csv")
    # data.to_csv("E:/code/addr_datat/618data.csv",encoding ='utf-8')
    values=[]
    for val in data["yy_data"]:
        values.append(int(val))
    values=np.array(values)
    mean=np.mean(values)
    # plt.plot([mean,mean],[0,0.2])显著影响抗菌药物占比变量汇总
    # plt.text(mean,0,"mean",ha = 'left',va = 'top')
    plt.legend()
    sns.distplot(values,color='black',label=u"原数据",hist=False)

    from ols_test import count_nums2,count_nums
    # result=count_nums2(["adrg_RE1","adrg_JE1","adrg_DJ1","adrg_FM3"],data)
    # result=count_nums2(["NL_老年","NL_童年","NL_青年","NL_中年","NL_少年"],data)
    # result=count_nums([["adrg_ES1","NL_青年"],["adrg_ES1","NL_老年"],["adrg_ES1","NL_童年"]],data)
    #result=count_nums2(["adrg_TT2","adrg_JE1","adrg_GB2","adrg_KC1"],data)
    result=count_nums2(["LYFS_2","LYFS_3","LYFS_4"],data)
    # result=count_nums([["adrg_BS1","NL_青年"],["adrg_FC3","NL_青年"],["adrg_JE1","NL_青年"]],data)
    color=['blue','red','green','yellow','gold']
    for i,key in enumerate(result):
        if len(result[key])<=1:
            print("too short")
            continue
        sns.distplot(result[key], color=color[i],label=key.decode('utf-8'), hist=False)
        mean=np.mean(result[key])
        #plt.plot([mean,mean],[0,0.3],linestyle='--',color=color[i])
        #plt.text(mean,0.2,"{}".format(round(mean,2)),ha = 'center',va = 'bottom')

    # _,yy2=count_nums(["NL_童年"],data)
    # _,yy3=count_nums(["NL_中年"],data)
    # _,yy4=count_nums(["NL_老年"],data)
    # _,yy5=count_nums(["NL_青年"],data)

if __name__=='__main__':
    out_tmp = "./tmp/search_159_out_tmp.csv"
    data=pd.read_csv(out_tmp)

    sns.distplot(data['KJYWZB'].values, color='black', label=u"原数据", hist=False)

    plt.legend()
    plt.xlim(0,50)
    plt.show()