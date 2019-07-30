from sklearn.feature_selection import VarianceThreshold

def var():
    """
    特征选择：VarianceThreshold
    对特征进行删选， 这里使用过滤器VarianceThreshold过滤掉一部分特征，减少特征数量，从而减少计算量。
    :return:
    """
    # threshold为超参数
    # 参数说明：
    # 默认把方差为0的一列删除掉，因为方差为0的一列特征值都是一样的
    var = VarianceThreshold(threshold=0.0)
    data = var.fit_transform([[0, 2, 0, 3],
        [0, 1, 4, 3],
        [0, 1, 1, 3]])
    print(data)

if __name__ == "__main__":
    var()

"""
运行结果如下：

[[2 0]
 [1 4]
 [1 1]]
"""