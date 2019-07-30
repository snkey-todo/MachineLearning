import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def file_len(fname):
    """
   获取文件的长度，也就是有多少行
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            # i为行，从0开始
            # l为每行的内容，
            pass
    # i还在内存中，可以直接调用
    return i + 1

def readcsv(filename):
    """
    读取csv文件
    """
    # 1、构造一个读取文件的队列
    filename_queue = tf.train.string_input_producer([filename])
    # 2、构建一个csv阅读器，去读取数据
    reader = tf.TextLineReader(skip_header_lines=0)
    # 开始读取数据，返回2个值，第一个值为行号，第二个值为内容
    _, csv_row = reader.read(filename_queue)
    # 3、文件解码setup CSV decoding
    record_defaults = [[0],[0],[0],[0],["None"]]
    c1,c2,c3,c4,label = tf.decode_csv(csv_row, record_defaults=record_defaults)
    # 转化为张量
    features = tf.stack([c1,c2,c3,c4])

    print("loading, " + str(file_length) + " line(s)\n")
    return features, label

# 要读取的数据文件
filename = "/Users/zhusheng/WorkSpace/Dataset/11-csv1/a1.csv"
# 获取文件长度
file_length = file_len(filename)
# 读取文件
features,label = readcsv(filename)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # 4、创建一个线程协调器，然后启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 5、循环读取
    for i in range(file_length):
        re_feature, re_label = sess.run([features, label])
        print(re_feature, re_label)

    # 6、读取结束后，关闭session、关闭线程
    coord.request_stop()
    coord.join(threads)

"""
运行结果如下：
loading, 10 line(s)

[   1   10  100 1000] b'train'
[   2   11  101 1001] b'bike'
[   3   12  102 1002] b'train'
[   4   13  103 1003] b'bike'
[   5   14  104 1004] b'subway'
[   6   15  105 1005] b'subway'
[   7   16  106 1006] b'bike'
[   8   17  107 1007] b'train'
[   9   18  108 1008] b'bike'
[  10   19  109 1009] b'subway'
"""