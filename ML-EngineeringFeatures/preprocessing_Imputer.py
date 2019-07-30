from sklearn.preprocessing import Imputer
import  numpy as np

def input():
    """
    缺失值处理
    :return:
    """
    # 参数说明：
    # missing_values指定缺失值，如果值为NaN就是缺失值
    # strategy指定计算缺失值的方式为求平均值，也就是说缺失值使用列的平均值代替
    # axis为按列计算
    input = Imputer(missing_values="NaN", strategy="mean", axis=0)

    data = [[1, 2],[np.nan, 3],[7, 6]]

    # 进行缺失值处理
    data = input.fit_transform(data)
    print(data)

if __name__ == "__main__":
    input()

"""
运行结果如下：
[[1. 2.]
 [4. 3.]
 [7. 6.]]
"""