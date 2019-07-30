from sklearn.feature_extraction import DictVectorizer

def dictvec():
    """
    字典数据抽取
    :return:
    """
    dict = DictVectorizer(sparse=False)
    data = [{'city': '北京','temperature':100},
            {'city': '上海','temperature':60},
            {'city': '深圳','temperature':30}]

    # 对字典数据进行特征抽取，转换为onehot编码
    result = dict.fit_transform(data)

    # 打印抽取的特征名称
    print(dict.get_feature_names())
    # 打印结果矩阵
    print(result)


if __name__ == "__main__":
    dictvec()


"""
运行结果如下：

['city=上海', 'city=北京', 'city=深圳', 'temperature']
[[  0.   1.   0. 100.]
 [  1.   0.   0.  60.]
 [  0.   0.   1.  30.]]
"""