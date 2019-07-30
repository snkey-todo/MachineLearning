import tensorflow as tf
import os

def csvread(filelist):
    """
    读取CSV文件
    :param filelist: 文件路径+名字的列表
    :return: 读取的内容
    """
    # 1、构造文件的队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2、构造csv阅读器读取队列数据（按一行）
    reader = tf.TextLineReader()
    # 读取数据，key为行号，value为每一行的内容
    key, value = reader.read(file_queue)

    # 3、对每行内容解码
    # record_defaults:指定每一个样本的每一列的类型，指定默认值[["None"], [4.0]]
    records = [["None"], [0]]
    example, label = tf.decode_csv(value, record_defaults=records)

    # 4、想要读取多个数据，就需要批处理
    # 一般capacity设置和batch_size一样，或者比它大。
    example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)

    return example_batch, label_batch

if __name__ == "__main__":
    # 需要读取的csv文件列表
    file_name = os.listdir("/Users/zhusheng/WorkSpace/Dataset/11-csv1/csv_list/")
    filelist = [os.path.join("/Users/zhusheng/WorkSpace/Dataset/11-csv1/csv_list/", file) for file in file_name]

    # 读取csv文件列表
    example_batch, label_batch = csvread(filelist)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        # 5、开启子线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # 6、打印获取的数据
        print(sess.run([example_batch, label_batch]))

        # 关闭线程
        coord.request_stop()
        coord.join(threads)

"""
运行结果如下：
[array([b'a1', b'a2', b'a3', b'a4', b'a5', b'a6', b'a7', b'a8', b'a9'],
      dtype=object), array([20, 21, 22, 23, 24, 25, 26, 27, 28], dtype=int32)]
"""