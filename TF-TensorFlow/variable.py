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