# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/16 09:58
 @Author  : hanzi5
 @Email   : hanzi5@yeah.net
 @File    : cart_classification.py
 @Software: PyCharm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import copy
from collections import Counter

matplotlib.rcParams['font.family'] = 'SimHei'  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 计算传入数据的Gini指数
def calcGini(dataSet,ycol):
    # dataSet=dataSet[list(dataSet[:,0]=='青年')]
    # 获得y中分类标签的唯一值
    y_lables = np.unique(dataSet[ycol])
    y_counts = len(dataSet)  # y总数据条数
    y_p = {}  # y中每一个分类的概率，字典初始化为空，y分类数是不定的，按字典存储更方便取值
    gini = 1.0
    for y_lable in y_lables:
        y_p[y_lable] = len(dataSet[dataSet[ycol] == y_lable]) / y_counts  # y中每一个分类的概率（其实就是频率）
        gini -= y_p[y_lable] ** 2
    return gini


# 计算Gini指数的另一种写法
def getGini(dataSet):
    # 计算基尼指数
    c = Counter(dataSet[:, -1])
    ret = 1 - sum([(val / dataSet[:, -1].shape[0]) ** 2 for val in c.values()])
    return ret


# 划分数据集
def splitDataSet(dataSet, col, value, types=1):
    # dataSet[list(dataSet[:,0]!='青年')]
    if types == 1:  # 使用此列特征中的value进行划分数据
        # subDataSet = dataSet[list(dataSet[:, i] == value)]  # 按照数据x第i列==value即可判断，不需要像《机器学习实战》书里写的那么复杂
        subDataSet=dataSet.loc[dataSet[col] == value]
        # subDataSet = np.array(subDataSet)  # 强制转换为array类型
    elif types == 2:  # 使用此列特征中的不等于value的进行划分数据
        subDataSet = dataSet.loc[dataSet[col] != value]  # 按照数据x第i列==value即可判断，不需要像《机器学习实战》书里写的那么复杂
        # subDataSet = np.array(subDataSet)  # 强制转换为array类型
    return subDataSet, len(subDataSet)


# 计算Gini指数，选择最好的特征划分数据集，即返回最佳特征下标及传入数据集各列的Gini指数
def chooseBestFeature(dataSet,ycol='', types='Gini'):
    numTotal = len(dataSet)  # 记录本数据集总条数
    numFeatures = len(dataSet.columns) - 1  # 最后一列为y，计算x特征列数
    bestFeature = -1  # 初始化参数，记录最优特征列i，下标从0开始
    columnFeaGini = {}  # 初始化参数，记录每一列x的每一种特征的基尼 Gini(D,A)
    x_cols=dataSet.columns.tolist()
    x_cols.remove(ycol)
    to_drop=[]
    for col in x_cols:  # 遍历所有x特征列
        # i=2
        prob = {}  # 按x列计算各个分类的概率
        featList = list(dataSet[col])  # 取这一列x中所有数据，转换为list类型
        prob = dict(Counter(featList))  # 使用Counter函数计算这一列x各特征数量
        if len(prob.keys())==1:
            to_drop.append(col)
            continue
        for value in prob.keys():  # 循环这一列的特征，计算H(D|A)
            # value='是'
            feaGini = 0.0
            bestFlag = 1.00001  # 对某一列x中，会出现x=是，y=是的特殊情况，这种情况下按“是”、“否”切分数据得到的Gini都一样，设置此参数将所有特征都乘以一个比1大一点点的值，但按某特征划分Gini为0时，设置为1
            subDataSet1, sublen1 = splitDataSet(dataSet, col, value, 1)  # 获取切分后的数据
            subDataSet2, sublen2 = splitDataSet(dataSet, col, value, 2)
            if (sublen1 / numTotal) * calcGini(subDataSet1,ycol) == 0:  # 判断按此特征划分Gini值是否为0（全部为一类）
                bestFlag = 1
            feaGini += (sublen1 / numTotal) * calcGini(subDataSet1,ycol) + (sublen2 / numTotal) * calcGini(subDataSet2,ycol)
            columnFeaGini['%s_%s' % (col, value)] = feaGini * bestFlag
    if len(columnFeaGini)==0:
        bestFeature='_'.join(x_cols)
    else: bestFeature = min(columnFeaGini, key=columnFeaGini.get)  # 找到最小的Gini指数益对应的数据列

    return bestFeature, to_drop


def createTree(dataSet,current_depth,max_depth,ycol='', types='Gini'):
    """
    输入：训练数据集D，特征集A，阈值ε
    输出：决策树T
    """

    y_lables_grp = dict(Counter(dataSet[ycol].values))  # 统计最优划分应该属于哪个i叶子节点“是”、“否”
    y_leaf = max(y_lables_grp, key=y_lables_grp.get)
    decisionTree = {'value': 0, 'node_class': 0}  # 构建树，以Gini指数最小的特征bestFeature为子节点
    decisionTree['node_class'] = y_leaf
    # 1、如果数据集D中的所有实例都属于同一类label（Ck），则T为单节点树，并将类label（Ck）作为该结点的类标记，返回T
    y_lables = np.unique(dataSet[ycol])
    if len(set(y_lables)) == 1 or current_depth >= max_depth or len(dataSet.columns) <= 2:
        decisionTree['value'] = None
        xcols=dataSet.columns.tolist()
        xcols.remove(ycol)
        decisionTree['feature'] = '_'.join(xcols)
        decisionTree['equal'] = None
        decisionTree['un_equal'] = None
        return decisionTree

    # 2、若特征集A=空，则T为单节点，并将数据集D中实例树最大的类label（Ck）作为该节点的类标记，返回T
    # if len(dataSet.columns) <= 2:
    #     labelCount = {}
    #     labelCount = dict(Counter(y_lables))
    #     return decisionTree

    # 3、否则，按CART算法就计算特征集A中各特征对数据集D的Gini，选择Gini指数最小的特征bestFeature（Ag）进行划分
    bestFeature, to_drop = chooseBestFeature(dataSet,ycol, types)
    if len(dataSet.columns) -2<=len(to_drop):
        decisionTree['value'] = None
        xcols = dataSet.columns.tolist()
        xcols.remove(ycol)
        decisionTree['feature'] = '_'.join(xcols)
        decisionTree['equal'] = None
        decisionTree['un_equal'] = None
        return decisionTree
    dataSet=dataSet.drop(to_drop,axis=1)
    # bestFeatureLable = bestFeature.split('_')[0]  # 最佳特征
    bestFeatureName,colVal=bestFeature.split('_')


    # del (features[int(bestFeature.split('_')[0])])  # 该特征已最为子节点使用，则删除，以便接下来继续构建子树

    # 使用beatFeature进行划分，划分产生2各节点，成树T，返回T
    try:
        y_lables_split = dataSet.loc[dataSet[bestFeatureName] == colVal][ycol]  # 获取按此划分后y数据列表
    except Exception:
        print(1)
    y_lables_grp = dict(Counter(y_lables_split.values))  # 统计最优划分应该属于哪个i叶子节点“是”、“否”
    y_leaf = max(y_lables_grp, key=y_lables_grp.get)  # 获得划分后出现概率最大的y分类
    print('feature:{},value:{},node_class:{},current_depth:{}'.format(bestFeatureName,colVal,y_leaf,current_depth))
    decisionTree['node_class'] = y_leaf
    decisionTree['value']=colVal
    decisionTree['feature']=bestFeatureName
    decisionTree['equal']=createTree(dataSet.loc[dataSet[bestFeatureName] == colVal].drop([bestFeatureName],axis=1),current_depth+1,max_depth,ycol,types)
    decisionTree['un_equal'] = createTree(dataSet.loc[dataSet[bestFeatureName] != colVal],current_depth+1,max_depth,ycol,types)


    # 4、删除此最优划分数据x列，使用其他x列数据，递归地调用步1-3，得到子树Ti，返回Ti
    # dataSetNew = np.delete(dataSet, int(bestFeature.split('_')[0]), axis=1)  # 删除此最优划分x列，使用剩余的x列进行数据划分
    # subFeatures = features[:]
    # 判断右枝类型，划分后的左右枝“是”、“否”是不一定的，所以这里进行判断
    return decisionTree


def print_tree(mytree,cond=[]):
    feature=mytree['feature']
    value=mytree['value']
    if mytree['equal']:

        tmp_cond=copy.deepcopy(cond)
        tmp_cond.append((feature, '{}'.format(value)))
        print_tree(mytree['equal'],tmp_cond)

    if mytree['un_equal']:
        tmp_cond = copy.deepcopy(cond)
        tmp_cond.append((feature,'not_{}'.format(value)))
        print_tree(mytree['un_equal'],tmp_cond)
    if not mytree['equal'] and not mytree['un_equal']:
        print('features:------------------------')
        print(cond)
        print('node_class:{}'.format(mytree['node_class']))

####以下是用来画图的完全复制的《机器学习实战》第三章的内容，不感兴趣的可以略过#############################################
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]  # myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    # depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]  # myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(myTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(myTree))
    plotTree.totalD = float(getTreeDepth(myTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(myTree, (0.5, 1.0), '')
    plt.show()


####画图结束，不感兴趣的可以略过#############################################

if __name__ == "__main__":
    # 读入数据，《统计学习方法》李航，P59，表5.1
    df_data = pd.read_csv('./tmp/out_cut.csv',dtype=str,encoding='gbk')
    # sdah=Counter(df_data['medicine'].values)
    df_data=df_data.drop(['KJYWZB'],axis=1)
    # vdj=df_data.loc[df_data['KJYWZB_CUT']==0]
    features = list(df_data.columns[1:-1])  # x的表头
    dataSet = df_data  # 数据处理为numpy.array类型，其实pandas.Dataframe类型更方便计算

    # 结果验证，计算结果与《统计学习方法》李航，P71页一致。
    # bestFeature, columnFeaGini = chooseBestFeature(dataSet,'KJYWZB_CUT', 'Gini')
    # print('\nbestFeature:', bestFeature, '\nGini(D,A):', columnFeaGini)

    dt_Gini = createTree(dataSet,0,15, 'KJYWZB_CUT', 'Gini')  # 建立决策树，CART分类树
    print_tree(dt_Gini)
    print('CART分类树：\n', dt_Gini)

    # 画出决策树
    createPlot(dt_Gini)
