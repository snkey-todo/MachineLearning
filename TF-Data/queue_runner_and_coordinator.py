import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 1、定义1个队列，可以放1000个数据
Q = tf.FIFOQueue(1000, tf.float32)

# 2、定义要做的事情
# 新建变量
var  = tf.Variable(0.0)
# 变量+1
data = tf.assign_add(var, tf.constant(1.0))
# 入队
en_q = Q.enqueue(data)

# 3、定义队列管理器op, 指定多少个子线程，子线程该干什么事
qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # 线程协调器、线程管理员
    coord = tf.train.Coordinator()
    # 真正开启子线程,去做那些事
    threads = qr.create_threads(sess, coord=coord, start=True)
    
    # 主线程，不断的去从队列中读取数据
    for i in range(300):
        print(sess.run(Q.dequeue()))
    # 回收子线程
    coord.request_stop()
    coord.join(threads)