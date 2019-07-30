from sklearn.preprocessing import MinMaxScaler

def minmax():
    """
    归一化处理

    归一化的最大值与最小值非常容易受异常点影响，所以一般我们不怎么使用归一化，而是标准化。
    :return:
    """
    mn  = MinMaxScaler(feature_range=(2,3))
    result = mn.fit_transform([[90,2,10,40],
        [60,4,15,45],
        [75,3,13,46]])
    print(result)

if __name__ == "__main__":
    minmax()

"""
当mn  = MinMaxScaler()
运行结果如下：
[[1.         0.         0.         0.        ]
 [0.         1.         1.         0.83333333]
 [0.5        0.5        0.6        1.        ]]

当  mn  = MinMaxScaler(feature_range=(2,3))
运行结果如下：
[[3.         2.         2.         2.        ]
 [2.         3.         3.         2.83333333]
 [2.5        2.5        2.6        3.        ]]
"""