import tensorflow as tf
import os

def pictureRead(filelist):
    # 1、构造文件队列
    queue = tf.train.string_input_producer(filelist)

    # 2、构造阅读器去读取图片内容（默认读取一张图片）
    reader = tf.WholeFileReader()
    key,value = reader.read(queue)

    # 3、解码
    images = tf.image.decode_jpeg(value)
    print("解码：",images.shape)

    # 4、统一图片大小
    images_resize = tf.image.resize_images(images, [200, 200])
    print("resize:",images_resize.shape)

    # 在批处理的时候要求所有数据形状必须定义
    # 如果是RGB图片，设置成[200, 200, 3]
    images_resize.set_shape([200,200,1])
    print("set_shape:",images_resize.shape)

    # 5、批处理,获得4D tensor， 第一个为样本数量
    images_batch = tf.train.batch([images_resize], batch_size=300, num_threads=1, capacity=300)
    print("批处理：", images_batch.shape)

    return images_batch


if __name__ == "__main__":
    # 图片列表路径
    dir_mstar = "/Users/zhusheng/WorkSpace/Dataset/8-MSTAR/MSTAR/EOC-data/train/2S1-b01/"

    filenames = os.listdir(dir_mstar)
    filelist = [os.path.join(dir_mstar, file) for file in filenames]
    #print("文件列表：",filelist)
    
    images_batch = pictureRead(filelist)

    with tf.Session() as sess:
        # 初始化变量
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        # 开启线程去读取图片
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        result = sess.run(images_batch)

        print("读取结果：", result[:1])
        
        # 回收线程
        coord.request_stop()
        coord.join(threads)

"""
运行结果如下：
解码： (?, ?, ?)
resize: (200, 200, ?)
set_shape: (200, 200, 1)
批处理： (300, 200, 200, 1)
读取结果： [[[[ 67.      ]
   [ 95.44    ]
   [ 61.819996]
   ...
   [ 31.029907]
   [ 31.139969]
   [ 41.      ]]

  [[105.71001 ]
   [134.7741  ]
   [104.985596]
   ...
   [ 55.43302 ]
   [ 59.40616 ]
   [ 73.39    ]]

  [[ 88.74    ]
   [120.260994]
   [100.8464  ]
   ...
   [ 96.37775 ]
   [101.66196 ]
   [115.06    ]]

  ...

  [[ 40.26001 ]
   [ 44.7946  ]
   [ 51.09242 ]
   ...
   [ 51.583595]
   [ 56.35912 ]
   [ 60.680176]]

  [[ 43.939987]
   [ 44.572002]
   [ 48.927616]
   ...
   [ 51.57547 ]
   [ 60.403603]
   [ 67.70003 ]]

  [[ 48.      ]
   [ 44.05    ]
   [ 44.16    ]
   ...
   [ 43.670044]
   [ 52.039978]
   [ 59.      ]]]]
"""