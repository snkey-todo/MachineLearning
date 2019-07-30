import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 1、定义队列,队列长度为3，数据类型为tf.float32
Q = tf.FIFOQueue(3, tf.float32)
# 往队列中放入数据
enq_many = Q.enqueue_many([[0.1, 0.2, 0.3],])

# 2、定义读数据、取数据的过程、取数据+1，入队列
# 出队列
de_q= Q.dequeue()
# 结果 + 1
data = de_q + 1
# 入队列
en_q = Q.enqueue(data)

with tf.Session() as sess:
    # 初始化队列
    sess.run(enq_many)

    # 处理数据，执行100次操作
    for i in range(100):
        # tensorflow的操作有依赖性
        sess.run(en_q)

    # 将队列中的数据都打印出来
    # 获取队列长度
    size = Q.size().eval()
    print("队列大小:", size)
    for i in range(size):
        print(sess.run(Q.dequeue()))

"""
运行结果如下：
队列大小: 3
33.2
33.3
34.1
"""