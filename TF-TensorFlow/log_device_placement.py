import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 创建tensor
a = tf.constant(5.0)    
b = tf.constant(6.0)

# 创建op
sum = tf.add(a, b) 

# 通过Session执行graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sum))


"""
运行结果如下：
Device mapping: no known devices.
Add: (Add): /job:localhost/replica:0/task:0/device:CPU:0
Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
Const_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
"""