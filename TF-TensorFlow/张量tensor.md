# 张量tensor

Tensor是一个类，包含了属性和常用函数，一个Tensor对象主要包含以下三个部分，如下所示：

```python
Tensor("Placeholder:0", shape=(2, 3), dtype=float32)
```

- 第一部分是`Tensor Name`，比如：’Constant’、‘Placeholder’、‘Variable’等，0没有什么实质性的含义，只是表示Tensor的顺序，当前是0，那么下一个Tensor就是1了；
- 第二部分是`Tensor Shape`, 也就是Tensor的形状，这里是[2,3]，2行3列。
- 第三部分是`Tensor Type`，也就是tensor张量的数据类型。

## tensor的形状

tensor的形状，我们在TensorFlow中使用阶乘表示。

|阶|数学实例|Python|例子|
|-|-|-|-|
|0|纯量|只有大小|s = 482|
|1|向量|有大小和方向|v = [1,2,3]|
|2|矩阵|数据表|m = [[1,2],[3,4],[5,6]]|
|3|3阶张量|数据立体|t = [[[1,2],[3,4]],[11,22],[33,44]]|
|n|n阶|n阶张量|...|

在Tensorflow中，矩阵[n,m]，表示n行m列，行表示特征数量，列表示样本数量。

## tensor的数据类型

|Python类型|描述|
|-|-|
|tf.float32|32位浮点数|
|tf.float64|64位浮点数|
|tf.int32|32位有符号整型|
|tf.int64|64位有符号整型|
|tf.int16|16位有符号整型|
|tf.int8|8位有符号整型|
|tf.uint8|8位无符号整型|
|tf.string|可变长度的字节数组，每一个张量元素都是一个字节数组|
|tf.bool|布尔型|
|tf.complex64|由两个32位浮点数组成的复数：实数和虚数|
|tf.qint32|用于量化Ops的32位有符号整型|
|tf.qint8|用于量化Ops的8位有符号整型|
|tf.quint8|用于量化Ops的8位无符号整型|

float32和float64表示浮点精度，但是实际并不会多分配内存，两者的使用效果差不多，我们常用float32。同理int32和int64也是这样。

## tensor属性

tensor对象有以下属性，我们可以通过tensor对象进行获取。

- graph：张量所在的图
- op:张量的op
- name:张量的名称
- shape:张量的形状

示例代码：

```python
import tensorflow as tf

a = tf.constant(1.0)

with tf.Session() as sess:
    print("graph-->", a.graph)
    print("op-->",a.op)
    print("name-->",a.name)
    print("shape-->",a.shape)
```

## placeholder占位符张量

我们可以通过`tf.placeholder()`来创建一个占位符张量，用于在运行图的时候，可以动态赋予数据。在Session中运行图的时候，我们通过`run(fetches, feed_dict=None, graph=None)`来动态赋予数据。

示例代码：

```python
import tensorflow as tf

plt = tf.placeholder(tf.float32, [2,3])

with tf.Session() as sess:
    sess.run(plt, feed_dict={
        plt: [[1, 2, 3],[3, 4, 5]]
    })
```

## 张量的操作

### 张量的动态形状和静态形状

静态形状
>张量的形状在整个图中都是固定不可变的，如果初始的张量中由不确定的形状（如`?`），我们可以通过`set_shape()`去设置,通过`get_shape()`去获取

示例代码：

```python
import tensorflow as tf

plt = tf.placeholder(tf.float32, [None,2])
# 列已经确定了，不能修改，行是None不确定可以修改
plt.set_shape([3,2])
print(plt.get_shape())
```

运行结果如下：

```bash
（3，2）
```

动态形状
>一种描述原始张量在执行过程中的一种形状，这个张量的形状在图的执行过程中是可以动态改变的。更新动态形状：`tf.reshape()`

示例代码：

```python
import tensorflow as tf

plt = tf.placeholder(tf.float32, [3,2])

# reshape重新创建一个张量
plt2 = tf.reshape(plt, [2,3])
print(plt2)
```

说明：reshape前后的张量数据个数肯定是不能变的。

### 随机张量

如果张量里的数据我们不确定的话，我们可以使用tensorflow提供的API创建随机张量。以下是从正态分布创建随机张量的例子：

```python
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32,seed=None, name=None)
```

说明：我们主要关注2个参数，mean表示数据平均值，也就是数学中的μ，stddev表示标准差，也就是数学中的σ。当μ=0，σ=1时，也就是标准正态分布。

### 张量类型转换

例如，如果我们的张量的dtype是tf.int32类型的，我们可以将其转换为tf.float32类型的。核心API：`tf.cast()`

示例代码：

```python
a = tf.constant(1.0)
b = tf.cast(a, tf.int32)
```

### 张量合并

我们可以将2个张量的数据进行合并，核心API是`tf.concat()`。

示例代码：

```python
import tensorflow as tf

b = [[1,2,3],[4,5,6]]
c = [[7,8,9],[10,11,12]]
# 张量合并
d = tf.concat([b,c], axis=0)

with tf.Session() as sess:
    print(d.eval())
```

运行结果如下：

```bash
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
```

说明：axis=0表示按行合并，axis=1表示按列合并。

