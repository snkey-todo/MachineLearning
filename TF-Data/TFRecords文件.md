# TFRecords文件

TFRecords是Tensorflow设计的一种`内置文件格式`，是一种`二进制文件`，它能更好的利用内存，方便进行数据的复制和移动。

TFRecords存储的文件格式为：`*.tfrecords`，文件写入的内容为：`Example协议块`。目的是为了将二进制数据和标签(训练的类别标签)数据存储在同一个文件中。

## 读写步骤分析

写入步骤：

1. 构造存储器
2. 构造每一个样本的Example
3. 写入序列化的Example

读取步骤：

1. 构造TFRecords阅读器
2. 解析Example
3. 转换格式，bytes解码

## 写入

参考`write_to_tfrecords.py`

### 建立TFRecord存储器

```python
# 1、建立TFRecords存储器
writer = tf.python_io.TFRecordWriter(path)
```

参数说明：

- path：TFRecords文件的路径，例如：`"./tmp/cifar.tfrecords"`,是一个以`.tfrecords`结尾的文件。
- return：返回一个文件写入器对象。

### 构造每个样本的Example协议块

```python
 # 构造一个样本的example
example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }))
```

参数说明：

- features：tf.train.Features类型的特征实例
- return：example格式协议块

### 写入序列化的Example

```python
writer.write(example.SerializeToString())
```

## 读取

参考`read_from_tfrecords.py`

### 构造TFRecords阅读器

```python
 # 1、构造TFRecords阅读器
reader = tf.TFRecordReader()
key, value = reader.read(file_queue)
```

### 解析Example

```python
# 2、解析Example
features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    })
```

example解析后包含了`image`部分和`label`部分，`image`部分需要解码。

### 转换格式，bytes解码

```python
# 3、转换格式，bytes解码
image = tf.decode_raw(features["image"], tf.uint8)
```
