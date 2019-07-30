# K-近邻算法(KNeighborsClassifier)

## K-近邻算法原理

### 定义

如果一个样本在`特征空间`中的`k个最相似的样本(即特征空间中最邻近的样本)属于某一个类别`，`则该样本也属于这个类别`。简单理解：也就是根据你的邻居来推断你的类别。

### 来源

KNN算法最早是由Cover和Hart提出的一种分类算法。

### 数学公式

`两个样本的距离`可以通过如下公式计算，又叫`欧式距离`。

比如说两个样本a(a1,a2,a3)，b(b1,b2,b3)，每个样本有三个特征。

![欧式距离公式](https://raw.githubusercontent.com/zhusheng/blog/master/ml/11.png)

## K-近邻算法优缺点

优点：

- 简单，易于理解，易于实现，无需估计参数，无需训练。

缺点：

- 懒惰算法，对测试样本分类时的计算量大，内存开销大。
- 必须指定K值，K值选择不当则分类精度不能保证。

k值取多大合适？
> 如果k值取很小，容易受异常点影响；
> 如果k值取很大，容易受数量波动。

使用场景
> 小数据场景，几千～几万样本，具体场景具体业务去测试。

性能问题
>性能问题非常明显，每一个新样本都需要和所有样本计算距离，如果样本非常大，计算非常耗时间。

## K-近邻算法案例，数学原理计算

样本数据如下：

![电影样本数据](https://raw.githubusercontent.com/zhusheng/blog/master/ml/12.png)

其中有一部电影的类别不确定，我们通过K-近邻算法来判断该电影是属于爱情片还是动作片。

![计算结果](https://raw.githubusercontent.com/zhusheng/blog/master/ml/13.png)

距离越近，属于那个类别的概率也就越大。

从上面我们可以看出，未知电影距离“Californial Man”、“He's not Reality into dues”、"Beautiful Women"比较近，而这三部电影都是爱情片，所以我们可以认为未知电影属于爱情片。

## K-近邻算法数据集

### facebook-v-predicting-check-ins

数据集`facebook-v-predicting-check-ins` 是facebook提供的一个数据集，也是Kaggle竞赛的数据集。数据集的目的：根据用户当前的位置，预测用户最可能签到的位置。

下载地址：`https://www.kaggle.com/c/facebook-v-predicting-check-ins/data`

本地数据集位置：`/Users/zhusheng/WorkSpace/Dataset/1-facebook-v-predicting-check-ins/facebook-v-predicting-check-ins/`
