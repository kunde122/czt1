'''
  整体：抗菌药物占比与其他影响因素的关系
            1.医院类型；`hospitalType`
            2.医院等级；grade
            3.ADRG组；
            4.实际住院时间；SJZYTS
            5.诊断记录次数（包含主要诊断和其它诊断合计16个诊断）；
            6.ADRG组内外科分类；
            7.离院方式；LYFS
            8.时间_年（如每年的不同的ADRG组的抗菌药物占比分布）；
            9.时间_月（某种的ADRG组病种的抗菌药物占比是否和季节性相关）；
            10.总费用和抗菌药物费用与抗菌药物占比的关系；
内科组：11.诊断中含有糖尿病的患者和不含糖尿病的对比有什么关系；
外科组：12.手术切口等级是否影响抗菌药物占比（第一个手术切口等级）；
           13.手术愈合类别是否影响抗菌药物占比（第一个手术愈合类别）；
           14.手术记录次数（7个手术）；
'''
import pandas as pd
import numpy as np
import regex as re
import os
ori_path="E:/code/addr_datat/search_159.xlsx"
path="E:/code/addr_datat/search_159_10w.xlsx"
path_out="E:/code/addr_datat/search_159_10w_out.xlsx"
path_proc="E:/code/addr_datat/search_159_10w_proc.xlsx"
path_proc_1="E:/code/addr_datat/search_159_10w_proc11.xlsx"
path_out11="E:/code/addr_datat/search_159_10w_out6-341.xlsx"
path_out_nonan="E:/code/addr_datat/search_159_10w_out_nonan.xlsx"
# import logging
# logging.basicConfig(level=logging.DEBUG,
#                 format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filename='myapp.log',
#                 filemode='w')
import sys
# log = open("mypr.log", "a")
# sys.stdout = log

def proc_1(in_path,out_path,limit_len=100000):
    data_excel = pd.read_excel(in_path)
    data_excel.drop(range(limit_len, len(data_excel)),inplace=True)
    drop_cols = []
    name1=re.compile('^JBDM')
    name2 = re.compile('^SSJCZBM')
    count_JBDM=[]
    count_SSJCZBM=[]
    yy_data1=[]
    for index, row in data_excel.iterrows():
        JBDM=0
        SSJCZBM=0
        yy=1
        for col_name in data_excel.columns:
            if re.match(name1,col_name):
                if not pd.isnull(row[col_name]):
                    JBDM+=1
            if re.match(name2,col_name):
                if not pd.isnull(row[col_name]):
                    SSJCZBM+=1
            if col_name=="ZFY":
                yy=yy/float(row[col_name])
            if col_name=="KJYWF":
                yy=yy*100*float(row[col_name])
        yy_data1.append(round(yy,2))
        if index%1000==0:
            print("processing line : {}".format(index))
        count_JBDM.append(JBDM)
        count_SSJCZBM.append(SSJCZBM)
    data_excel["JBCOUNT"]=count_JBDM
    data_excel["SSCOUNT"] = count_SSJCZBM
    data_excel["yy_data"] = yy_data1
    drop_mcols(data_excel,'^JBDM\d')
    drop_mcols(data_excel, '^SSJCZBM\d')

    data_excel.to_excel(out_path)
# proc_1(ori_path,path_proc_1)

def proc():
    data_excel=pd.read_excel(path)
    count=len(data_excel)
    data_excel=data_excel.dropna(axis=1,thresh=count*0.5)
    col_count=len(data_excel.columns)
    data_excel=data_excel.dropna(axis=0,thresh=col_count*0.5)
    data_excel.to_excel(path_out)

def fun1():
    data_excel=pd.read_excel(path_out)

    #手术及操作编码1
    adrg=data_excel["SSJCZBM1"]
    #总费用
    ZFY=data_excel["ZFY"]
    #抗菌药物费
    KJYWF=data_excel["KJYWF"]
    res=[]
    for i,(zfy,kjywf) in enumerate(zip(ZFY,KJYWF)):
        tmp=round(100*kjywf/zfy,2)
        res.append(tmp)
        print("{}/{}:{:.2f}%".format(kjywf,zfy,tmp))
    # data_excel["yy_data"]=res
    # data_excel.to_excel(path_out)
    import numpy as np
    res=np.array(res)



def anova_test(data_path,result_path):
    data=pd.read_excel(data_path)
    import xlsxwriter
    workbook = xlsxwriter.Workbook(result_path)
    worksheet = workbook.add_worksheet("sheet1")
    bold_red_style = workbook.add_format({'bold': True,'color':'red', 'align' : 'center'})
    bold_style=workbook.add_format({'bold': True, 'align' : 'center'})
    center_style = workbook.add_format({ 'align' : 'center'})
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    # col_name_list=["adrg","LYFS","NL","JBDM","JBDM1","SSJCZBM1","SJZYTS"]
    col_name_list = ["SSCOUNT"]
    yy_name = data.columns[-1]
    row=0
    res={}
    for col_name in data.columns[:-1]:
        col = 0
        data_tmp = data.dropna(subset=[col_name])
        count = set()
        for item in data_tmp[col_name]:
            count.add(item)
        # if col_name not in col_name_list:
        #     continue
        # if len(count)>2000:
        #     continue
        print("col_name:{},class_num:{}".format(col_name,len(count)))
        worksheet.write(row, col, (u"因素：{},水平数：{}".format(col_name,len(count))), bold_red_style)
        # row+=1

        try:
            model = ols('{}~ C({})'.format(yy_name, col_name), data_tmp).fit()
            anovaResults = anova_lm(model)
            res[col_name]=anovaResults['F'][0]
            print(anovaResults)
        except Exception:
            continue
        col=1
        for col_n in anovaResults.columns:
            worksheet.write(row, col, col_n,bold_style)
            col+=1

        row += 1
        for index, row_ in anovaResults.iterrows():
            col=0
            worksheet.write(row, col, index,center_style)
            col+=1
            for cl_res, cl_name in enumerate(anovaResults.columns):
                if not pd.isnull(row_[cl_name]):
                    worksheet.write(row, col, row_[cl_name],center_style)
                col += 1
            row += 1
        row += 1
    res = sorted(res.items(), key=lambda x: x[1], reverse=True)
    for key in res:
        print(key)
    worksheet.set_column("A:A", 50)
    workbook.close()

#手术编码小数点后保留一位
def proc_data_1(in_path,out_path):
    data_excel = pd.read_excel(in_path)

    # name_mode=re.compile("^JBDM")
    name_mode = re.compile("^JBDM|^SSJCZBM1")
    mode=re.compile("(?<=\.\w).*")
    for col_name in data_excel.columns:
        if re.match(name_mode,col_name):
            for i,value in enumerate(data_excel[col_name]):
                if pd.isnull(value) :
                    continue
                print(value)
                value=re.sub(mode,'',value)
                print(value)
                data_excel[col_name][i]=value
    data_excel.to_excel(out_path)
# proc_data_1(path_proc_1,path_proc_1)

def proc_data_2(in_path,out_path):
    # 对年龄进行划分
    data_excel = pd.read_excel(in_path)
    res=np.array(data_excel["NL"])
    res_list=pd.cut(res, [-1,6,17,40,65,120],labels=[u"童年",u"少年",u"青年",u"中年",u"老年"])
    data_excel["NL"]=res_list
    data_excel.to_excel(out_path)
# proc_data_2(path_proc_1,path_proc_1)
# proc_data_1(path_out,path_proc)
col_mode=re.compile("C\(|\).*")
def get_n(p_name):
    res=re.sub(ext_mode,'',p_name)
    return res
# get_n("C(adrg)[T.AE1]")

def anova_test_nonan(in_path,out_path):
    data = pd.read_excel(in_path)
    data.dropna(axis=0, how='any',inplace=True)
    # data.to_excel("E:/code/addr_datat/search_159_10w_proc22.xlsx")
    print("length of row:{}".format(len(data)))
    # var="adrg+NL+SJZYTS+LYFS+JBDM+createTime+QKDJ1+QKYHLB1+JBCOUNT+SSCOUNT"
    xx_var="C(adrg)+C(NL)+SJZYTS+C(LYFS)+C(JBDM)+C(createTime)+C(QKDJ1)+C(QKYHLB1)+C(JBCOUNT)+C(SSCOUNT)"
    yy_var="yy_data"

    from statsmodels.formula.api import ols
    model=ols("{}~{}".format(yy_var,xx_var),data).fit()
    parameter = pd.DataFrame(model.tvalues)
    pvalues_param=pd.DataFrame(model.pvalues)
    f=lambda name:re.sub(col_mode,'',name)
    max_tvalue={}
    for name, group in parameter.groupby(f):
        max_tvalue[name]=max(abs(group.values.max()),abs(group.values.min()))
        # print(name)
    # for id,id_name in enumerate(parameter._index):
    #     print(parameter[id_name])


    # columns=parameter.columns
    # from statsmodels.stats.outliers_influence import summary_table
    # st, data, ss2 = summary_table(model, alpha=0.05)

    max_tvalue=sorted(max_tvalue.items(),key=lambda x:x[1],reverse=True)
    for key in max_tvalue:
        print(key)

    print(model.summary())
    print(max_tvalue)
# anova_test_nonan(path_proc_1,'')

def anova_test_find(in_path,out_dir="E:/code/addr_datat/spss_code"):
    in_path="E:/code/addr_datat/search_159_10w_proc222.xlsx"
    data = pd.read_excel(in_path)
    # data.dropna(axis=0, how='any',inplace=True)
    # data.to_excel("E:/code/addr_datat/search_159_10w_proc22.xlsx")
    print("length of row:{}".format(len(data)))
    # var="adrg+NL+SJZYTS+LYFS+JBDM+createTime+QKDJ1+QKYHLB1+JBCOUNT+SSCOUNT"
    # xx_var="C(adrg)+C(NL)+C(SJZYTS)+C(LYFS)+C(JBDM)+C(createTime)+C(QKDJ1)+C(QKYHLB1)+C(JBCOUNT)+C(SSCOUNT)"
    xx_var = "C(adrg)+C(NL)+C(SJZYTS)+C(LYFS)+C(JBDM)+C(createTime)+C(QKDJ1)+C(QKYHLB1)+C(JBCOUNT)+C(SSCOUNT)"
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

    from itertools import groupby
    f = lambda name: re.sub(col_mode, '', name)
    na_mode=re.compile('.*\[T.|\]')
    def ff(item):
        col=re.sub(col_mode, '', item[0])
        name=re.sub(na_mode, '', item[0])
        return col,name

    result={}
    for it in max_tvalue.items():
        col, name=ff(it)
        if col not in result:
            result[col]=[name]
        else:result[col].append(name)
    from spss_code import create_dumy2
    # group_dic=groupby(max_tvalue.items(),key=ff)
    for name,group in result.items():
        out_f=os.path.join(out_dir,'_{}'.format(name))
        print(out_f)
        create_dumy2(name,group,out_f)




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

def anova_test_ana(in_path,out_dir=""):
    #in_path="E:/code/addr_datat/search_159_10w_proc222.xlsx"
    in_path="E:/code/addr_datat/618data.xlsx"
    data = pd.read_excel(in_path)
    print("length of row:{}".format(len(data)))
    # var="adrg+NL+SJZYTS+LYFS+JBDM+createTime+QKDJ1+QKYHLB1+JBCOUNT+SSCOUNT"
    # xx_var="C(adrg)+C(NL)+C(SJZYTS)+C(LYFS)+C(JBDM)+C(createTime)+C(QKDJ1)+C(QKYHLB1)+C(JBCOUNT)+C(SSCOUNT)"
    xx_var = "C(adrg)+C(NL)+SJZYTS+C(LYFS)+C(QKDJ1)+C(QKYHLB1)+C(JBCOUNT)+C(SSCOUNT)"
    yy_var="KJYWZB"
    # xx_var=''
    # for col in data.columns:
    #     if '_' in col and col !=yy_var:
    #         if len(xx_var)==0:
    #             xx_var += col
    #         else:xx_var+='+'+col
    # xx_var+='+SJZYTS'


    from statsmodels.formula.api import ols
    model=ols("{}~{}".format(yy_var,xx_var),data).fit()
    # dt=pd.DataFrame(model)
    cnx=model.summary()
    print(model.summary())

    pvalues_param=pd.DataFrame(model.pvalues)
    indexs=pvalues_param.index
    max_tvalue={}
    for ind,value in zip(pvalues_param.index,pvalues_param.values):
        if abs(value[0])<0.05:
            max_tvalue[ind]=value[0]
            print(ind,value)

    from itertools import groupby
    f = lambda name: re.sub(col_mode, '', name)
    na_mode=re.compile('.*\[T.|\]')
    def ff(item):
        col=re.sub(col_mode, '', item[0])
        name=re.sub(na_mode, '', item[0])
        return col,name

    result={}
    for it in max_tvalue.items():
        col, name=ff(it)

        if col not in result:
            result[col]=[name]
        else:result[col].append(name)
    for key in result:
        for name in result[key]:
            print(name)
    return
    from spss_code import create_dumy2
    # group_dic=groupby(max_tvalue.items(),key=ff)
    # for name,group in result.items():
    #     out_f=os.path.join(out_dir,'_{}'.format(name))
    #     print(out_f)
    #     create_dumy2(name,group,out_f)



    # for ind in indexs:
    #     if pvalues_param[ind]<0.05:
    #         max_tvalue[ind]=pvalues_param[ind,0]




    # columns=parameter.columns
    # from statsmodels.stats.outliers_influence import summary_table
    # st, data, ss2 = summary_table(model, alpha=0.05)

    max_tvalue=sorted(max_tvalue.items(),key=lambda x:x[1],reverse=True)
    for key in max_tvalue:
        print(key,max_tvalue[key])

    print(model.summary())
    print(max_tvalue)

anova_test_ana(path_proc_1)
# anova_test(path_proc_1,path_out_nonan)

{"data":[
    {"cid":"123","csdz":"容县县底镇县底圩石龙路50号","xzz":"容县县底镇县底圩石龙路50号","hkdz":"容县县底镇县底圩石龙路50号"},
    ...
]}

{   "flag":0,
    "msgErr":"",
    "data":[
        {"cid":"123","csdz_s":{"PROVINCE":"广西壮族自治区","CITY":"玉林市","COUNTY":"容县"},
         "xzz_s": {"PROVINCE": "广西壮族自治区", "CITY": "玉林市", "COUNTY": "容县"},
         "hkdz_s": {"PROVINCE": "广西壮族自治区", "CITY": "玉林市", "COUNTY": "容县"},
         },
    ...
]}


def linearR(data_path):
    data = pd.read_excel(data_path)
    data.dropna(axis=0, how='any', inplace=True)
    print("length of row:{}".format(len(data)))

    from sklearn import linear_model
    model=linear_model.LinearRegression()
    yy=data._iloc[:,-1].values
    xx=data._iloc[:,:-1].values
    res=model.fit(xx,yy)
    print(res)
linearR(path_proc_1)
 # xx=np.array(xx)
import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')
# plt.hist(res)
# 创建画图窗口
fig = plt.figure()
# 将画图窗口分成1行1列，选择第一块区域作子图
ax1 = fig.add_subplot(1, 1, 1)
# 设置标题
ax1.set_title('Result Analysis')
# 设置横坐标名称
ax1.set_xlabel('gamma-value')
# 设置纵坐标名称
ax1.set_ylabel('R-value')
# 画散点图
ax1.scatter(xx, res, s=5, c='k', marker='.')
#
# 调整横坐标的上下界
# plt.xlim(xmax=5, xmin=0)
plt.show()
adrg_set=set()
for adr in adrg:
    if pd.isnull(adr):
        continue
    adrg_set.add(adr)
print(len(adrg_set))




gds=data_excel.isnull().sum()
for col in data_excel.columns:
    print(gds[col])



for zfy,kjywf in zip(ZFY,KJYWF):
    tmp=100*kjywf/zfy
    print("{}/{}:{:.2f}%".format(kjywf,zfy,tmp))

