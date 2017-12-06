# -*- coding:utf-8 -*-

import scipy as sp
import matplotlib.pyplot as plt
import tight

#取数据
data = sp.genfromtxt('./data/web_traffic.tsv', delimiter="\t")
#print data

#数据预处理和数据清洗
x = data[:,0]
y = data[:,1]

#print sp.sum(sp.isnan(y))

#挑选y值合法的选项
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]


# 显示数据
plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")

plt.xticks( [w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])
plt.autoscale(tight==True)
plt.grid()
plt.show()