# 读取csv文件

## 构建文件队列

```python
filename_queue = tf.train.string_input_producer([filename])
```

该函数返回一个文件队列，便于使用阅读器去读取。

## 构建一个csv阅读器，去读取数据

```python
reader = tf.TextLineReader(skip_header_lines=0)
```

该函数返回一个阅读器对象。

csv文件是一行一行的读取，`skip_header_lines=0`表示是否跳过行首，也就是说行首是否也读取。

## 开始读取数据

```python
_, csv_row = reader.read(filename_queue)
```

该函数返回2个值，第一个值为行号，第二个值为内容。一行一行的读取，所以我们需要在后面使用循环去读取。

## 文件解码

```python
record_defaults = [[0],[0],[0],[0],["None"]]
c1,c2,c3,c4,label = tf.decode_csv(csv_row, record_defaults=record_defaults)
```

`record_defaults`表示每一个字段的数据类型。如果是整数类型，写`0`；如果是浮点类型，写`0.0`；如果是字符串类型，写`"None"`、`"null"`等。

该函数返回的是没一个字段的值。

## 创建线程循环读取

```python
# 4、创建一个线程协调器，然后启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 5、循环读取
    for i in range(file_length):
        re_feature, re_label = sess.run([feature, label])
        print(re_feature, re_label)

    # 6、读取结束后，关闭session、关闭线程
    coord.request_stop()
    coord.join(threads)
```

注意：`sess.run`运行参数和接收参数一定要不一样，否则第二次运行的时候会报错“Can not convert a ndarray into a Tensor or Operation.”，算是踩的一个大坑。
