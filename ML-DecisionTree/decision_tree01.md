# 决策树算法流程

## 读取数据

```python
 titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
```

## 数据预处理

（1）理解问题，确定我们要干什么

首先，我们确定我们的分析指标，以什么为特征值，以什么为目标值。

这里我们以plass(舱位级别)、age(年龄)、sex(性别)作为特征值，survived(生存)作为目标值，分析这些特征的人群生存率，通过决策树算法训练出一个树状模型。

当有一个新的样本过来，我们通过决策树来判断这个人是否生存。

```python
 x = titanic[['pclass', 'age', 'sex']]
 y = titanic['survived']
```

（2）缺失值处理

在上面的数据集中，我们的age是存在缺失值的，所以，我们需要做缺失值处理。

在下面的代码中，我们将缺失值使用`年龄平均值`进行替补,`inplace=True`表示替换。

```python
x['age'].fillna(x['age'].mean(), inplace=True)
```

（3）数据集分割

然后，我们进行数据集分割，我们将数据集分割为训练集和测试集，训练集75%，测试集25%。

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
```

## 特征处理

首先我们进行特征抽取，特征抽取的目的是将数据转换为计算机能处理的数据。

`船舱级别pclass`有3个类别，分别为：1st|2nd|3rd；`sex`有2个类别，分别是：male|female，`age`本身就是数值型的。

针对类别数据，我们可以将其转换为字典类型，然后转换为one-hot编码。

```python
dict = DictVectorizer(sparse=False)
x_train = dict.fit_transform(x_train.to_dict(orient="records"))
x_test = dict.transform(x_test.to_dict(orient="records"))
```

`x_train.to_dict(orient="records")`用于将数据集转换为字典数据，然后使用字典特征提取器`DictVectorizer`来提取特征。

## 算法训练和预测

我们选择决策树算法来进行分类。

首先，我们创建一个决策树算法对象。

```python
dtc = DecisionTreeClassifier(max_depth=5)
```

`max_depth`为超参数，表示决策树的层级，这里我们设置`max_depth=5`表示决策树一共5层。

超参数的选择会影响最终的结果，所以在解决实际问题的时候，我们都需要优化超参数。

然后我们使用训练集来训练决策树，并使用测试集来测试样本的准确率，也就是模型的表现效果。

```python
dtc.fit(x_train, y_train)
print("预测准确率:", dtc.score(x_test, y_test))
```

