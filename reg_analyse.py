import pandas as pd
import random
# import regex as re
from sklearn.datasets import load_boston
import numpy as np
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import trange

class Analyse():
    def __init__(self,data_path=''):
        if data_path:
            self.data=pd.read_csv(data_path)

    def creat_dummy(self,df,cols=['']):
        for col in cols:
            dummy_ranks = pd.get_dummies(df[col], prefix=col)
            df=df.join(dummy_ranks.ix[:,1:])
        return df.drop(cols, axis=1)

    def creat_dummy1(self,df,cols=['']):
        for col in cols:
            dummy_ranks = pd.get_dummies(df[col], prefix=col)
            df=df.join(dummy_ranks)
        return df.drop(cols, axis=1)

    def stepwise_selection(self,X, y,
                           initial_list=[],
                           threshold_in=0.01,
                           threshold_out=0.05,
                           verbose=True):
        """ Perform a forward-backward feature selection
        based on p-value from statsmodels.api.OLS
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            initial_list - list of features to start with (column names of X)
            threshold_in - include a feature if its p-value < threshold_in
            threshold_out - exclude a feature if its p-value > threshold_out
            verbose - whether to print the sequence of inclusions and exclusions
        Returns: list of selected features
        Always set threshold_in < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
        """
        included = list(initial_list)

        while True:
            changed = False
            # forward step
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.argmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

            # backward step
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()  # null if pvalues is empty
            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.argmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
            if not changed:
                break
        return included

    def anova_test_ana(self,csv_data, out_dir=""):
        print("length of row:{}".format(len(csv_data)))
        # var="adrg+NL+SJZYTS+LYFS+JBDM+createTime+QKDJ1+QKYHLB1+JBCOUNT+SSCOUNT"
        # xx_var="C(adrg)+C(NL)+C(SJZYTS)+C(LYFS)+C(JBDM)+C(createTime)+C(QKDJ1)+C(QKYHLB1)+C(JBCOUNT)+C(SSCOUNT)"
        xx_var = "C(adrg)+C(NL)+SJZYTS+C(LYFS)+C(QKDJ1)+C(QKYHLB1)+C(JBCOUNT)+C(SSCOUNT)"
        yy_var = "KJYWZB"

        from statsmodels.formula.api import ols
        model = ols("{}~{}".format(yy_var, xx_var), csv_data).fit()

        print(model.summary())

        pvalues_param = pd.DataFrame(model.pvalues)
        indexs = pvalues_param.index
        max_tvalue = {}
        for ind, value in zip(pvalues_param.index, pvalues_param.values):
            if abs(value[0]) < 0.05:
                max_tvalue[ind] = value[0]
                print(ind, value)

    def convert_to_num(self,df,cols=[]):
        if len(cols)==0:
            cols=df.columns
        from sklearn import preprocessing

        map_save={}
        for col in cols:
            la = preprocessing.LabelEncoder()
            # gh=print(sum(data[col].isnull()))
            la.fit(df[col].astype(str))
            df[col]=la.transform(df[col].astype(str))
            map_save[col]=la
        return df,map_save


    def tree_regressor(self,X_train,y_train):
        from sklearn.tree import DecisionTreeClassifier,export_graphviz
        from sklearn.model_selection import train_test_split

        # X_train, map_save = self.convert_to_num(X_train, cols=['adrg', 'NL'])
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=33)
        # import pydotplus
        from sklearn.externals.six import StringIO
        X_train = self.creat_dummy1(X_train, cols=['adrg','QKDJ1','LYFS','QKYHLB1'])
        X_train,map_save=self.convert_to_num(X_train,cols=['JBCOUNT','SSCOUNT'])

        clf = DecisionTreeClassifier(splitter='best')
        clf.fit(X_train, y_train)
        print("train score:", clf.score(X_train, y_train))
        # print("test score:", clf.score(X_test, y_test))
        #dot -Tpdf tree2.dot -o out2.pdf
        with open("./tmp/tree2.dot", 'w') as f:
            f = export_graphviz(clf, feature_names=X_train.columns.tolist(),max_depth=5, out_file=f)  # 输出结果至文件
        # from graphviz import Source
        # graph = Source(export_graphviz(clf, out_file='./tmp/tree.dot', feature_names=X_train.columns.tolist()))
        # graph.format = 'png'
        # graph.render('dt', view=True);
        # dot_data = StringIO()
        # export_graphviz(clf, feature_names=X_train.columns.tolist(), out_file=dot_data,max_depth=30)
        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # graph.write_pdf("./tmp/tree.pdf")
        return clf

        # print("test score:", clf.score(X_test, y_test))

    def cross_table(self,data):
        # 交叉表分析，丢弃不显著的列
        excludes = []
        from  scipy.stats import chi2_contingency
        for col in data.columns:
            if col == 'KJYWZB' or col == "KJYWZB_CUT":
                continue
            rw = pd.crosstab(data['KJYWZB_CUT'], data[col])
            # print(rw.head())
            x2 = chi2_contingency(rw)
            print(col, x2[1])
            if x2[1] > 0.05:
                excludes.append(col)
        if len(excludes) > 0:
            data = data.drop(excludes, axis=1)
        return data

    def find_k(self,values):
        SSE=[]

        values=values.reshape([len(values), 1])
        for i in trange(1,20):
            est=KMeans(n_clusters=i)
            est.fit(values)
            SSE.append(est.inertia_)
        X=range(1,20)
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.plot(X,SSE,'o-')
        plt.show()

    def group_by_cluster(self,df,col,k_num):
        # 利用kmeans对因变量分组
        est = KMeans(n_clusters=k_num)
        values = df[col].values
        est.fit(values.reshape([len(values), 1]))

        labels = np.array(est.labels_)
        bins = [-1]
        for i in range(k_num):
            count = sum(labels == i)
            print("class {},count:{}".format(i, count))
            bins.append(np.max(values[labels == i]))
        bins.sort()
        print(bins)
        df['KJYWZB_CUT'] = pd.cut(df[y_col], bins=bins).values.codes
        return df
    def check_nan(self,df):
        for col in df.columns:
            assert sum(df[col].isnull()) == 0
            print('{}:null count {}'.format(col,sum(df[col].isnull())))
    def print_values_count(self,df):
        for col in df.columns:
            print(df[col].value_counts())

if __name__=="__main__":
    analyse=Analyse()
    out_tmp = "./tmp/origin_tmp.csv"
    out_tmp2 = "./tmp/origin_tmp2.csv"
    out_res1 = "./tmp/origin_res1.csv"
    out_res2 = "./tmp/origin_res2.csv"
    data=pd.read_csv(out_tmp2,encoding='gbk')
    # analyse.print_values_count(data)
    y_col = "KJYWZB"
    # data = analyse.group_by_cluster(data, y_col, 5)
    # dfss=pd.qcut(data['KJYWZB'].values,20)
    # data['KJYWZB_CUT']=list(pd.qcut(data['KJYWZB'].values,20).codes)
    # rw = analyse.cross_table(data)
    # rw1 = np.array(rw)
    # analyse.check_nan(data)
    # print(sum(data['KJYWZB'].isnull()))

    # cou=pd.value_counts(data['SJZYTS'])
    # print(cou)
    # analyse.find_k(data['KJYWZB'].values)


    # data=analyse.group_by_cluster(data,y_col,5)
    data = data.drop(['createTime'], axis=1)
    # data['NL']=list(pd.cut(data['NL'].values,bins=[-1, 6, 17, 40, 65, 130] , labels=[u"童年", u"少年", u"青年", u"中年", u"老年"]))
    # data=data.ix[:2000,[0,1,2,3]]
    # data['LYFS']=data['LYFS'].astype(int)
    # data.to_csv('./tmp/out_cut.csv', index=False)
    # dgr=pd.qcut(data[y_col],20)

    cols = data.columns.tolist()

    cols.remove('KJYWZB')
    # cols.remove('KJYWZB_CUT')
    cols.remove('medicine')
    # cols.remove('LYFS')
    # cols.remove('QKDJ1')
    # cols.remove('QKYHLB1')
    cols.remove('grade')
    # X = data[cols]
    # y = data.KJYWZB_CUT
    # cla=analyse.tree_regressor(X,y)
    # data=analyse.cross_table(data)




    '''
    rw=pd.crosstab(data.NL,data.adrg)
    rw1=np.array(rw)
    
    
    print(rw1)
    '''
    #线性回归分析
    #创建虚拟变量['adrg','NL','LYFS','QKDJ1','QKYHLB1']
    data=analyse.creat_dummy(data,cols=['adrg','LYFS','QKYHLB1','QKDJ1','grade','medicine'])
    # data.to_csv('./tmp/out_cut.csv',index=False)
    # data=data.astype(float)
    # analyse.check_nan(data)
    gd=data.index.is_unique
    print(gd)
    cols = data.columns.tolist()
    # cols.remove('adrg')
    cols.remove('KJYWZB')
    yy_var='KJYWZB'
    xx_var='+'.join(cols)
    # indexs=random.sample(range(len(data)),200000)
    # indexs=np.array(indexs)
    # data=data.loc[indexs,:]
    data = data.loc[100000:300000, :]
    # analyse.check_nan(data)
    print(len(data))
    model = ols('{}~{}'.format(yy_var, xx_var),data).fit()#sm.add_constant(
    dwa=model.summary().tables[1]
    result=pd.DataFrame(dwa[1:])
    result.columns=dwa[0]
    # result.to_csv(out_res1,index=False)
    print(model.summary())

    drop_cols=[col for col in model.pvalues._index if float(model.pvalues[col]) >= 0.05 or pd.isnull(model.pvalues[col])]

    cols = data.columns.tolist()
    cols.remove('KJYWZB')
    # cols.remove('adrg')
    for col in drop_cols:
        cols.remove(col)
    yy_var = 'KJYWZB'
    xx_var = '+'.join(cols)

    model = ols('{}~{}'.format(yy_var, xx_var), data).fit()
    print(model.summary())
    dwa=model.summary().tables[1]
    result=pd.DataFrame(dwa[1:])
    result.columns=dwa[0]
    result.to_csv(out_res2,index=False)
    # result = analyse.stepwise_selection(X, y)

    print('resulting features:')
    # print(result)
