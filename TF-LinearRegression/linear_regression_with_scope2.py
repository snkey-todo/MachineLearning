import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def mymigration():
    
    with tf.variable_scope("data"):
        x = tf.random_normal([100,1], mean=1.75, stddev=0.5, name="x")
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        weight = tf.Variable(tf.random_normal([1,1], mean=0.0, stddev=1.0, name="w"))
        bias = tf.Variable(0.0, name="b")
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_true- y_predict))

    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 1、收集tensor
    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weights", weight)
    # 2、合并变量，写入事件文件
    merged = tf.summary.merge_all()
    
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        # 建立事件文件
        filewritter = tf.summary.FileWriter('logs/', graph=sess.graph)

        # 运行优化
        print("随机最先初始化的权重和偏置：权重为:%f, 偏置为:%f" % (weight.eval(), bias.eval()))
        for i in range(200):
            sess.run(train_op)

            # 3、运行合并的变量
            summary = sess.run(merged)
            # 4、写入事件文件
            filewritter.add_summary(summary, i)
            
            print("第%d次优化的权重和偏置：权重为:%f, 偏置为:%f" % (i, weight.eval(), bias.eval()))
    return None

if __name__ == "__main__":
    mymigration()