# -*- coding:utf-8 -*-

COLOUR_FIGURE = False

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

# 取出数据
data = load_iris()
features = data['data']
feature_names = data['feature_names']
species = data['target_names'][data['target']]

# 选出其它两类的数据
setosa = (species == 'setosa')
features = features[~setosa]
species = species[~setosa]
virginica = (species == 'virginica')


# 计算分界线
best_acc = -1.0
for fi in xrange(features.shape[1]):
    thresh = features[:, fi].copy()
    thresh.sort()

    for t in thresh:
        pred = (features[:, fi] > t)
        acc = (pred == virginica).mean()

        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
print("Best acc: %f" % best_acc)
print("Best fi: %f" % best_fi)
print("Best t: %f" % best_t)

'''
best_t结果为1.6,测试发现1.75分界线能到更好的分界效果，具体参考图
'''

t = 1.75
p0,p1 = 3,2

if COLOUR_FIGURE:
    area1c = (1.,.8,.8)
    area2c = (.8,.8,1.)
else:
    area1c = (1.,1,1)
    area2c = (.7,.7,.7)

'''
features[:,p0]表示取出第4个特征数值
'''
x0,x1 =[features[:,p0].min()*.9,    features[:,p0].max()*1.1]
y0,y1 =[features[:,p1].min()*.9,    features[:,p1].max()*1.1]

'''
指定颜色填充区域
'''
plt.fill_between([t,x1],[y0,y0],[y1,y1],color=area2c)
plt.fill_between([x0,t],[y0,y0],[y1,y1],color=area1c)

plt.plot([t,t],[y0,y1],'k--',lw=2)
plt.plot([t-.1,t-.1],[y0,y1],'k:',lw=2)

'''
c=b ,表示蓝色

marker:0显示为O,x显示为x，>显示为三角形，目前我知道的就三种
'''
plt.scatter(features[virginica,p0], features[virginica,p1], c='b', marker='o')
plt.scatter(features[~virginica,p0], features[~virginica,p1], c='r', marker='x')
plt.ylim(y0,y1)
plt.xlim(x0,x1)
plt.xlabel(feature_names[p0])
plt.ylabel(feature_names[p1])
plt.savefig('./MLChapter2/images/1400_02_02.png')

