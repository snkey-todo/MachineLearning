# TensorBoard入门

tensorboard是tensorflow框架为我们提供的可视化web界面，我们可以在web界面中看到我们的tensorflow graph，并查看图的结构、图的运行结果、监视的一些过程数据、一些参数的优化过程等等。

示例代码：

```python
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

x =  tf.Variable(tf.random_normal([1,1], mean=0.0, stddev=1.0))
w = tf.constant(5.0)
b = tf.constant(2.0)

y = tf.matmul(x, [[w]]) + b

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # 将图结构写入事件文件（evnets file）中,我们需要指定目录
    filewritter = tf.summary.FileWriter('logs/', sess.graph)
    print(sess.run(y))
```

当我们执行完上面的指令后，会在”logs/”目录下生成一个`事件文件(events file)`，这个`目录需要我们先创建好`。

每执行一次程序都会生成一个events file，这个events file中就包含了我们的图数据。tensorborad默认会加载最新的，覆盖之前的events file。

然后，我们在命令行执行如下代码，就可以通过浏览器访问`http://127.0.0.1:6006`

```bash
tensorboard --logdir='logs/'
```

如果我们有多个events file,运行上述指令会提示：”Overwriting the metagraph with the newest event.”也就是说tensorboard会覆盖之前的事件文件。

总结：在tensorboard可视化中，events file是核心，我们evnets file生成成功了，自然就可以使用tensorboard进行可视化了。
