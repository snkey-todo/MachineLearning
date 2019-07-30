from sklearn.preprocessing import StandardScaler

def standard():
    """
    标准化
    :return:
    """
    std = StandardScaler()
    data = std.fit_transform([[ 1., -1., 3.],
        [ 2., 4., 2.],
        [ 4., 6., -1.]]
        )
    print(data)

if __name__ == "__main__":
    standard()


"""
运行结果如下：
[[-1.06904497 -1.35873244  0.98058068]
 [-0.26726124  0.33968311  0.39223227]
 [ 1.33630621  1.01904933 -1.37281295]]
"""