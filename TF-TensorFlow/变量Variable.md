# 变量Variable

变量也是一种OP，是一种特殊的张量，能够进行存储持久化。

## 创建Variable

API：`tf.Variable(initial_value=None,name=None)`

参数说明：

- assign(value)：为变量分配一个新值返回新值
- eval(session=None)：计算并返回此变量的值
- name：属性表示变量名字

示例代码：

```python
import tensorflow as tf
import os
os.environ['TP_CPP_LOG_LEVEL'] = '2'

# 创建变量
val = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0))

# 变量初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行变量初始化op
    sess.run(init_op)

    # 输出变量
    print(sess.run(val))
```

我们`在会话前必须做变量的初始化成一个op`，然后`在会话中首先运行这个初始化变量`。
