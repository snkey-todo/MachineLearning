import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

base_dir = "/Users/zhusheng/WorkSpace/MachineLearning"
print(base_dir)
events_file_dir = base_dir + "/logs/"

x =  tf.Variable(tf.random_normal([1,1], mean=0.0, stddev=1.0))
w = tf.constant(5.0)
b = tf.constant(2.0)

y = tf.matmul(x, [[w]]) + b

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # 将图结构写入事件文件（evnets file）中,我们需要指定目录
    filewritter = tf.summary.FileWriter(events_file_dir, sess.graph)

    print(sess.run(y))