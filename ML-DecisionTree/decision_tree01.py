# -*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_graphviz
import pandas as pd

def tree_decision():
    """
    决策树预测泰坦尼克号生死
    :return:
    """
    # 读取数据
    titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 数据预处理
    # 选择分析的特征值和目标值
    x = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']
    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征抽取
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    x_test = dict.transform(x_test.to_dict(orient="records"))
    # 打印特征名称
    print("get_feature_names:",  dict.get_feature_names())
    print(x_train)

    # 使用决策树进行分类
    dtc = DecisionTreeClassifier(max_depth=30)
    dtc.fit(x_train, y_train)
    print("预测准确率:", dtc.score(x_test, y_test))

    # 导出决策树的结构
    export_graphviz(dtc, out_file='./tree.dot', feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male'])


if __name__ == "__main__":
    tree_decision()