#coding:utf-8
import numpy as np
# from docx import Document
import pandas as pd
# from statsmodels.formula.api import ols

path='E:/code/addr_datat/adrgadrg.txt'

def sort_key(line):
    num=line.strip().split()[1]
    num=float(num)
    return num
import pandas as pd

def save_as_docs(out_path,rows_data={},col_names=[]):
    document=Document()
    row_len=len(rows_data)
    col_len=len(col_names)
    table = document.add_table(rows=1, cols=col_len+1)  # 插入表格
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    fix_cols = {}
    for id,col in enumerate(col_names):
        hdr_cells[id+1].text = col
        fix_cols[col] = id + 1
    for row_key in rows_data.keys():
        row_cells = table.add_row().cells
        row_cells[0].text=row_key
        for key in rows_data[row_key]:
            data=rows_data[row_key][key]
            row_cells[fix_cols[key]].text=str(data)
    rown=row_key.split('_')[0]
    coln=col_names[0].split('_')[0]

    document.save('{}-{}.docx'.format(rown,coln))  # 保存文档

def count_nums(pairs,data=pd.DataFrame()):
    res = {}
    search={}
    for pair in pairs:
        tmp={}
        for  item in pair:
            col, val = item.strip().split('_')
            tmp[col]=val
        key='-'.join(tmp.values())
        search[key]=tmp
        res[key]=[]


    for i,row in data.iterrows():
        for key in search:
            item_search=search[key]
            find = True
            for col in item_search:
                if str(row[col])!=item_search[col]:
                    find=False
                    break
                else:
                    print(row[col],item_search[col])
            if find:
                res[key].append(row["yy_data"])
    return res

def count_nums2(pairs,data=pd.DataFrame()):
    res={}
    search={}
    for ori in pairs:
        col,val=ori.strip().split('_')
        search[ori]=(col,val)
        res[ori]=[]
    for i,row in data.iterrows():
        print("num:{}".format(i))
        for ori in pairs:
            col, val=search[ori]
            if str(row[col])==val:
                res[ori].append(row["yy_data"])
    return res

def save_result(out_path,rows_data,col_names):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(out_path)
    worksheet = workbook.add_worksheet("sheet1")
    fix_cols={}
    for i,key in enumerate(col_names):
        fix_cols[key]=i+1
        worksheet.write(0, i+1, key)


    for id,row_key in enumerate(rows_data):
        worksheet.write(id + 1, 0, row_key)
        for key in rows_data[row_key]:
            data=rows_data[row_key][key]
            worksheet.write(id+1,fix_cols[key],str(data))
    workbook.close()

if __name__=="__main__":
    in_path="E:/code/addr_datat/618data.xlsx"
    data=pd.read_excel(in_path)
    print(data["yy_data"].values.mean())
    cols=[['adrg_FP1','NL_童年'],['adrg_FP1','NL_青年'],['adrg_FP1','NL_老年'],['adrg_RE1','NL_童年'],['adrg_RE1','NL_青年'],['adrg_RE1','NL_老年']]

    adrg_pos=['adrg_JE1','adrg_RE1','adrg_ES1','adrg_ST1']
    adrg_neg=['adrg_BS1','adrg_FC3','adrg_FP1']
    NL_pos=['NL_青年']
    NL_neg=['NL_童年','NL_老年']
    JBCOUNT_pos=["JBCOUNT_2","JBCOUNT_3"]
    QKDJ1_pos=["QKDJ1_4","QKDJ1_3"]
    SSCOUNT_neg=["SSCOUNT_2","SSCOUNT_3","SSCOUNT_4"]

    col2=[]
    adrg_col=adrg_pos+adrg_neg
    NL_col=NL_pos+NL_neg
    rows={}
    col_names=[]

    row_c=adrg_col
    col_c=JBCOUNT_pos
    for c1 in row_c:
        tmp={}
        for c2 in col_c:
            tmp[c2]=0
            col2.append([c1,c2])
        rows[c1] =tmp
    for c2 in col_c:
        col_names.append(c2)
    for col in col2:
        count,  vals = count_nums(col, data)
        rows[col[0]][col[1]]=[count,round(np.mean(vals),2)]
        print(col[0],col[1]+': ',count, np.mean(vals))

    save_as_docs("E:/code/addr_datat/count.xlsx",rows,col_names)







# lines=[]
# with open(path) as fin:
#     lines=fin.readlines()
# lines=sorted(lines,key=sort_key,reverse=True)
#
# with open('E:/code/addr_datat/adrga.txt','w') as fout:
#     for line in lines:
#         fout.write(line)











'''
inter=['1','2','3','4']
np.random.seed(15)
col1=np.random.rand(10)
col2=np.random.rand(10)
col3=np.random.choice(inter,10)
data={"col1":col1,"col2":col2,"col3":col3}
data_tmp=pd.DataFrame(data)
model = ols('col1~ C(col3)+col2', data_tmp).fit()
print(model.params)

col31=np.array(col3=='1',dtype=int)
col32=np.array(col3=='2',dtype=int)
col33=np.array(col3=='3',dtype=int)
col34=np.array(col3=='4',dtype=int)

data2={"col1":col1,"col2":col2,"col3":col3,'col31':col31,'col32':col32,'col33':col33}
data_tmp2=pd.DataFrame(data2)
model = ols('col1~ col34+col32+col33+col2', data_tmp2).fit()
print(model.params)
'''