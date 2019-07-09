#coding:utf-8
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import regex as re
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


def dist():
    out_tmp = "./tmp/origin_tmp.csv"
    data = pd.read_csv(out_tmp, encoding='gbk')
    cols = ['ET1', 'FN2', 'FM2']
    vals=['-3.99','-19.27','-20.25']
    cols = [0, 1, 4]
    color = ['blue', 'red', 'green', 'yellow', 'gold']
    sns.distplot(data['KJYWZB'].values, color='black', label=u"原数据", hist=False)

    # tmp = data.loc[data['adrg'] == 'TR2']
    tmp=data
    for i, col in enumerate(cols):
        values = tmp.loc[tmp['NL'] == col]['KJYWZB'].values
        sns.distplot(values, color=color[i], label='{}-{}岁'.format(int(col)*10,int(col+1)*10), hist=False)
        # sns.distplot(values, color=color[i], label='{}({})'.format(col,vals[i]), hist=False)
    plt.legend()
    plt.xlim(0, 50)
    plt.show()

def cont():
    dict={'NL':'年龄','SJZYTS':'实际住院天数','JBCOUNT':'疾病编码个数','SSCOUNT':'手术记录次数'}
    out_tmp = "./tmp/origin_tmp.csv"
    out_tmp="./tmp/origin_tmp11.csv"
    col='JBCOUNT'
    data = pd.read_csv(out_tmp, encoding='gbk')
    group=data['KJYWZB'].groupby(data[col]).mean()

    color = ['blue', 'red', 'green', 'yellow', 'gold']
    fig = plt.figure()
    ax1=fig.add_axes((0.1,0.1,0.8,0.8))
    ax1.set_title("{}与抗菌药物占比的关系".format(dict[col]))
    ax1.plot(group._index.values,group.values , linestyle='--', color=color[0])

    start=0
    end=4
    prop = sum((data[col] >= start) & (data[col] <= end)) / float(len(data))

    ax1.plot(group._index.values[start:end], group.values[start:end],label='{}%'.format(round(100*prop,0)), linestyle='--', color=color[2])
    plt.legend()
    # plt.fill_between(group._index.values,group.values,where=(group._index.values >= start) & (group._index.values <= end))
    ax1.set_xlabel('{}(回归系数-0.31)'.format(dict[col]))
    ax1.set_ylabel('抗菌药物占比(%)')
    ax1.grid(color=color[0], linestyle='--', linewidth=1, alpha=0.3)
    plt.xticks(np.arange(0, 20, 1))

    left,bottom,width,height=(0.5,0.6,0.25,0.25)

    ax2=fig.add_axes((left,bottom,width,height))
    ax2.set_title('{}频率分布'.format(dict[col]))
    sns.distplot(data[col].values, color=color[1], hist=False)
    ax2.grid(color='r',linestyle='--',linewidth=1,alpha=0.3)
    plt.xticks(np.arange(0,20,1))
    ax2.set_xlabel(dict[col])
    ax2.set_ylabel('频率')
    plt.xlim(0,20)

    plt.show()

def disp():
    out_tmp = "./tmp/origin_res2.csv"
    data=pd.read_csv(out_tmp)
    cols=data.columns.tolist()
    cols[0]='index'
    data.columns=cols  #groupby  key
    data_tmp = data[data['index'].map(lambda item: re.search('adrg', item) != None)]
    data_tmp.sort_values(by='coef',inplace=True)

    out_tmp = "./tmp/origin_tmp.csv"
    data = pd.read_csv(out_tmp, encoding='gbk')
    group=data['KJYWZB'].groupby(data['adrg']).median()
    group1 = data['KJYWZB'].groupby(data['adrg']).mean()
    res=[]
    resm=[]
    for name in data_tmp['index']:
        name=name.split('_')[1]
        res.append(group[name])
        resm.append(group1[name])
    fig = plt.subplot(211)

    # ax1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    fig.plot(range(len(res)),res)

    plt.ylabel('中位数')


    fig = plt.subplot(212)
    fig.plot(range(len(resm)), resm)
    # plt.xlabel('adrg按系数升序排列')
    plt.xlabel('adrg按系数升序排列')
    plt.ylabel('均值')
    plt.show()
    fjs=0


if __name__=='__main__':
    # dist()
    cont()
    # disp()