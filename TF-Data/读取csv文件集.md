# 读取csv文件集

相比读取csv单个文件，多了一个步骤，就是`批处理`。

## 构建文件队列

```python
file_queue = tf.train.string_input_producer(filelist)
```

## 构建一个csv阅读器，去读取数据

```python
reader = tf.TextLineReader(skip_header_lines=0)
```

## 开始读取数据

```python
_, csv_row = reader.read(filename_queue)
```

## 内容解码

```python
records = [["None"], [0]]
example, label = tf.decode_csv(value, record_defaults=records)
```

## 文件批处理

```python
# 4、想要读取多个数据，就需要批处理
# 一般capacity设置和batch_size一样，或者比它大。
example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)
```

函数说明：利用一个tensor的列表或字典来获取一个batch数据

```python
tf.train.batch(
    tensors,
    batch_size,
    num_threads=1,
    capacity=32,
    enqueue_many=False,
    shapes=None,
    dynamic_pad=False,
    allow_smaller_final_batch=False,
    shared_name=None,
    name=None
)
```

参数说明：

- tensors：一个列表或字典的tensor用来进行入队
- batch_size：设置每次从队列中获取出队数据的数量
- num_threads：用来控制入队tensors线程的数量，如果num_threads大于1，则batch操作将是非确定性的，输出的batch可能会乱序
- capacity：一个整数，用来设置队列中元素的最大数量
- enqueue_many：在tensors中的tensor是否是单个样本
- shapes：可选，每个样本的shape，默认是tensors的shape
- dynamic_pad：Boolean值.允许输入变量的shape，出队后会自动填补维度，来保持与batch内的shapes相同
- allow_samller_final_batch：可选，Boolean值，如果为True队列中的样本数量小于batch_size时，出队的数量会以最终遗留下来的样本进行出队，如果为Flalse，小于batch_size的样本不会做出队处理
- shared_name：可选，通过设置该参数，可以对多个会话共享队列
- name：可选，操作的名字
