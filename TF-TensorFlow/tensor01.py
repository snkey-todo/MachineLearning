import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

b = [[1,2,3],[4,5,6]]
c = [[7,8,9],[10,11,12]]
# 张量合并
d = tf.concat([b,c], axis=0)

with tf.Session() as sess:
    print(d.eval())