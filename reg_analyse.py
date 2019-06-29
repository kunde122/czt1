import pandas as pd
import regex as re
from sklearn.datasets import load_boston
import numpy as np
import statsmodels.api as sm
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
        import pydotplus
        from sklearn.externals.six import StringIO
        X_train,map_save=self.convert_to_num(X_train,cols=['adrg','NL'])

        clf = DecisionTreeClassifier(splitter='best')
        clf.fit(X_train, y_train)

        print("train score:", clf.score(X_train, y_train))

        with open("./tmp/tree1.dot", 'w') as f:
            f = export_graphviz(clf, feature_names=X_train.columns.tolist(),max_depth=3, out_file=f)  # 输出结果至文件
        from graphviz import Source
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
            if x2[1] > 0.05:
                print(col, x2[1])
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
        df['KJYWZB_CUT'] = pd.cut(df[y_col], bins=bins).values.labels
        return df


if __name__=="__main__":
    analyse=Analyse()
    out_tmp = "./tmp/search_159_out_tmp.csv"
    data=pd.read_csv(out_tmp)
    # analyse.find_k(data['KJYWZB'].values)
    y_col = "KJYWZB"

    data=analyse.group_by_cluster(data,y_col,5)
    data = data.drop(['createTime'], axis=1)

    data['NL']=list(pd.cut(data['NL'].values,bins=[-1, 6, 17, 40, 65, 130] , labels=[u"童年", u"少年", u"青年", u"中年", u"老年"]))
    # data=data.ix[:2000,[0,1,2,3]]
    # data['LYFS']=data['LYFS'].astype(int)
    data.to_csv('./tmp/out_cut.csv', index=False)
    # dgr=pd.qcut(data[y_col],20)

    cols = data.columns.tolist()

    cols.remove('KJYWZB')
    cols.remove('KJYWZB_CUT')
    X = data[cols]
    y = data.KJYWZB_CUT
    cla=analyse.tree_regressor(X,y)
    data=analyse.cross_table(data)




    '''
    rw=pd.crosstab(data.NL,data.adrg)
    rw1=np.array(rw)
    
    
    print(rw1)
    '''
    #线性回归分析
    #创建虚拟变量
    data=analyse.creat_dummy(data,cols=['adrg','NL','LYFS','QKDJ1','QKYHLB1','JBCOUNT','SSCOUNT','grade'])
    data.to_csv('./tmp/out_cut.csv',index=False)

    model = sm.OLS(y, sm.add_constant(pd.DataFrame(X))).fit()
    print(model.summary())
    result = analyse.stepwise_selection(X, y)

    print('resulting features:')
    print(result)
