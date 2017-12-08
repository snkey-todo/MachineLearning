# -*- coding:utf-8 -*-
import os

import scipy as sp
import matplotlib.pyplot as plt
import tight

# read the data from the file
data = sp.genfromtxt('./data/web_traffic.tsv', delimiter="\t")
#print data

colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']

#数据预处理
x = data[:,0]
y = data[:,1]

# 和数据清洗：挑选y值合法的选项
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]


# plot input data and save the results as a picture
def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    plt.clf()
    # 指定坐标为x，y
    plt.scatter(x, y, s=10)
    # 指定标题
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    # 设置横轴的记号
    plt.xticks(
        [w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname)
# 查看误差
def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)

# first look at the data
#plot_models(x, y, None, os.path.join("./MLChapter1/images", "1400_01_01.png"))


# create and plot models
# polyfit创建模型类型，我们使用我们提炼出来的数据组x和y创建了一阶数据模型，也就是一条直线。
fp1, res, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
#print("Model parameters: %s" % fp1)


f1 = sp.poly1d(fp1)
f2 = sp.poly1d(sp.polyfit(x, y, 2))
f3 = sp.poly1d(sp.polyfit(x, y, 3))
f10 = sp.poly1d(sp.polyfit(x, y, 10))
f100 = sp.poly1d(sp.polyfit(x, y, 100))


#plot_models(x, y, [f1], os.path.join("./MLChapter1/images", "1400_01_01-1.png"))

'''
绘制模型
1400_01_02.png绘制了函数f1、1400_01_03.png同时绘制函数f1和f2进行对比、1400_01_04.png绘制了多个函数进行对比
'''
#plot_models(x, y, [f1], os.path.join("./MLChapter1/images", "1400_01_02.png"))
#plot_models(x, y, [f1, f2], os.path.join("./MLChapter1/images", "1400_01_03.png"))
#plot_models(x, y, [f1, f2, f3, f10, f100], os.path.join("./MLChapter1/images", "1400_01_04.png"))
'''
查看误差
print error(f1, x, y)
print error(f2, x, y)
print error(f3, x, y)
print error(f10, x, y)
print error(f100, x, y)
'''

'''
重新训练数据，将数据分为两批处理，所有有两个数据集(xa,ya)\(xb,yb)
'''
inflection = int(3.5 * 7 * 24)
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

# 生成新的训练模型
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

# 绘图
# plot_models(x, y, [fa, fb], os.path.join("./MLChapter1/images", "1400_01_05.png"))

# 查看误差
print("Error inflection= %f" % error(fa, xa, ya))
print("Error inflection= %f" % error(fb, xb, yb))
