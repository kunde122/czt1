import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from numpy import random
import regex as re
from tqdm import tqdm
import collections

class DataClean():
    def __init__(self,data_path=''):
        if data_path:
            self.data=pd.read_csv(data_path)

    def fill_nan(self):
        pass
    def random_fill(self,df,col,select=300):
        count = collections.Counter(df[col].values).most_common(select)
        str_list = []
        prob = []
        for srr, c in count:
            if pd.isnull(srr):
                continue
            str_list.append(srr)
            prob.append(c)
        prob = normalize(prob, norm='l1')[0]
        for i, adrg in enumerate(df[col]):
            if pd.isnull(adrg):
                df[col].values[i] = str(random.choice(str_list, p=prob))
        return df

    def add_JBDM1_SS(self,df,pre_str=['']):
        print(len(df['adrg'].values))
        res=np.zeros((len(df),len(pre_str)))
        col_matchs=[]
        for strr in pre_str:
            col_matchs.append(re.compile(strr))
        with tqdm(desc='count nums: ', total=len(df)) as pbar:
            for row_index, row in df.iterrows():
                for col_name in df.columns:
                    for id,match in enumerate(col_matchs):
                        if re.search(match, col_name):
                            if not pd.isnull(row[col_name]):
                                res[row_index][id]+=1
                pbar.set_description('Processing line {}'.format(row_index))
                pbar.update()

            # if row_index % 1000 == 0:
            #     print("processing line : {}".format(row_index))
        pbar.close()
        #creat cols
        for id,strr in enumerate(pre_str):
            df[strr[1:3]+'COUNT']=res[:,id]
        return df

    def delete_cols(self,df,cols=['']):
        name_mode=re.compile('|'.join(cols))
        cols = []
        for col_name in df.columns:
            if re.match(name_mode, col_name):
                cols.append(col_name)
        df=df.drop(cols, axis=1)
        return df

    def add_KJYWZB(self,df):
        yy_data1=[]

        for row_index, row in df.iterrows():
            if row_index%1000==0:
                print("processing:line {}".format(row_index))
            yy = 1
            for col_name in df.columns:
                if col_name=="ZFY":
                    yy=yy/float(row[col_name])
                if col_name=="KJYWF":
                    yy=yy*100*float(row[col_name])
            yy_data1.append(round(yy,2))
        df['KJYWZB']=yy_data1
        return df

    def cut_NL(self,df,col='',bins=[],labels=['']):
        '''
        分割数据
        :param df:
        :param col:
        :param bins: 切分点，如[-1, 6, 17, 40, 65, 120]
        :param labels: 每一段的显示标签，如[u"童年", u"少年", u"青年", u"中年", u"老年"]
        :return:
        '''
        values=df[col].values
        res_list = pd.cut(values,bins , labels=labels)
        # split = pd.qcut(values, nums)
        df[col]=res_list
        return df

    def cut_yy(self,df,col='',num=20):
        values = df[col].values
        split = pd.qcut(values, num)
        print(split.value_counts())
        df[col] = split.labels
        return df

    def print_category(self,df):
        for col_name in df.columns:
            count = collections.Counter(df[col_name].astype(str).values)
            print(col_name,len(count))
            print(count.most_common(3))
            print('--------------------------------------------------')
mode=re.compile('QY$')

def process_origin(data):
    # 删除adrg组的缺失值和QY结尾歧义值
    data = data.dropna(subset=['adrg']).reset_index(drop=True)
    data = data[data['adrg'].map(lambda item: re.search(mode, item) == None)]
    print("length:{}".format(len(data)))
    print('adrg null_count', sum(data['adrg'].isnull()))

    # 填充QKDJ1和QKYHLB1缺失值为0
    data["QKDJ1"] = data["QKDJ1"].fillna(0)
    print('QKDJ1 null_count', sum(data['QKDJ1'].isnull()))
    data["QKYHLB1"] = data["QKYHLB1"].fillna(0)
    print('QKYHLB1 null_count', sum(data['QKYHLB1'].isnull()))

    data["QKYHLB1"].replace({'甲': 0, '乙': 1, '丙': 3, '其他': 9}, inplace=True)
    data["QKYHLB1"] = data["QKYHLB1"].astype(float).astype(int)
    print(pd.value_counts(data['QKYHLB1']))
    data["QKDJ1"].replace({'Ⅱ': 3, 'Ⅱ/ 甲': 3, 'Ⅰ/ 甲': 2, 'I': 2, 'Ⅰ': 2, '0类': 1, 'Ⅲ': 4,
                           'Ⅲ/ 甲': 4, 'II': 3, 'Ⅱ/ 乙': 3, 'Ⅲ/ 丙': 4, 'III': 4,
                           'zero': 1, 'Ⅰ/ 丙': 2, 'Ⅰ/ 乙': 2, 'Ⅱ/ 丙': 3, 'Ⅲ/ 乙': 4}, inplace=True)
    data = data[data['QKDJ1'] != '郝国君']
    print(pd.value_counts(data['QKDJ1']))
    data["QKDJ1"] = data["QKDJ1"].astype(float).astype(int)
    print(pd.value_counts(data['QKDJ1']))

if __name__=="__main__":
    ori_path = "./tmp/origin.csv"
    out_put = "./tmp/origin_out.csv"
    out_tmp = "./tmp/origin_tmp.csv"
    out_tmp1 = "./tmp/origin_tmp11.csv"
    out_tmp2 = "./tmp/origin_tmp2.csv"
    out_res="./tmp/res.csv"
    data = pd.read_csv(out_tmp1, encoding='gbk')


    # res=data['KJYWZB'].groupby(data['adrg']).agg(['mean','count','max','min'])
    # res.to_csv(out_res,encoding='gbk')
    # print(data['KJYWZB'].groupby(data['adrg']).agg(['mean','count','max','min']))

    print(data['adrg'].value_counts())

    # data.drop(['Unnamed: 0'], axis=1, inplace=True)
    # data.drop(['Unnamed: 0'],axis=1,inplace=True)
    # data.to_csv(ori_path,index = False)
    print("length:{}".format(len(data)))
    dataClean = DataClean()
    if False:
        # 对年龄分组
        data = dataClean.cut_NL(data, col='NL', bins=[-1,10 , 20, 30, 40, 50,60,70,80,90,100,110,130],labels=list(range(12)))
        data.to_csv(out_tmp2, index=False, encoding='gbk')
    # dataClean.print_category(data)
    if False:
        # 添加JBCOUNT和SSMCOUNT
        data = dataClean.add_JBDM1_SS(data, ['^JBDM', '^SSJCZBM'])
        data = dataClean.delete_cols(data, ['^JBDM', '^SSJCZBM'])
        data.to_csv(out_tmp1, index=False,encoding='gbk')

    if 'KJYWZB' not in data.columns.tolist():
        # 添加KJYWZB列
        # data = pd.read_csv(out_tmp, encoding='gbk')
        data=data[data['ZFY']>0]
        print(len(data))
        data = dataClean.add_KJYWZB(data)
        data = dataClean.delete_cols(data, ['ZFY', 'KJYWF'])
        data.to_csv(out_tmp1, index=False,encoding='gbk')

    # 对年龄分组
    data = dataClean.cut_NL(data, col='NL', bins=[-1, 6, 17, 40, 65, 120],
                            labels=[u"童年", u"少年", u"青年", u"中年", u"老年"])

