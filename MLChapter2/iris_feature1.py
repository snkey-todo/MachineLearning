# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# 直接调用sklearn.datasets数据集中的方法取得数据,它是一个JSON字符串
data = load_iris()

# 数据特征值
features = data['data']

# 数据特征值名称
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
feature_names = data['feature_names']

#该数组是数据源的真实种类是什么，0-setosa，1-versicolor，2-virginica
target = data['target']



# 有4个特征值，使用二维坐标显示就有6中组合。
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

# 对原始数据可视化
#plt.savefig('./images/1400_02_01.png')


'''******************第一次二分*******************'''
# 训练模型
labels = data['target_names'][data['target']]
#print labels

# features是一个二维数组，我们取出其中的petal length属性获得一个新的一维数组
plength = features[:,2]
#print features
#print plength

# 查看setosa花的petal length范围
is_setosa = (labels == 'setosa')
#print('Maximum of setosa: {0}.'.format(plength[is_setosa].max()))
#print('Minimum of others: {0}.'.format(plength[~is_setosa].min()))

'''
if plength < 2:
    print 'Iris Setosa'
else:
    print 'Iris versicolor or Iris virginica'
'''

'''******************第二次二分*******************'''
features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')