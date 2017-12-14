# -*- coding:utf-8 -*-

from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# 直接调用sklearn.datasets数据集中的方法取得数据,它是一个JSON字符串,我将其打印出来拷贝到"./data/json.txt"文件中
data = load_iris()
#print data


# 数据特征值及名称，我们需要根据这4个特征来分类
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
feature_names = data['feature_names']
features = data['data']

# 三种类型的花，0-setosa，1-versicolor，2-virginica
target = data['target']

# 有4个特征值，采用（x,y）坐标显示的话就有6中形式，我们将已有的数据可视化为6个图
pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

for i,(p0,p1) in enumerate(pairs):
    plt.subplot(2,3,i+1)
    for t,marker,c in zip(range(3),">ox","rgb"):
        plt.scatter(features[target == t,p0], features[target == t,p1], marker=marker, c=c)

    # x轴为sepal length，y轴为sepal width
    plt.xlabel(feature_names[p0])
    plt.ylabel(feature_names[p1])
    plt.xticks([])
    plt.yticks([])

plt.savefig('./MLChapter2/images/1400_02_01.png')