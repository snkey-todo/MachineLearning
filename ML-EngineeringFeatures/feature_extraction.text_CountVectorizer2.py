from sklearn.feature_extraction.text import CountVectorizer

def countvec():
    """
    对文本进行特征值化
    :return:
    """
    countvec = CountVectorizer()
    data = ["生命短暂，我喜欢python，我喜欢python",
        "生命太长，我不喜欢python"]

    result = countvec.fit_transform(data)

    print(countvec.get_feature_names())
    print(result.toarray())

if __name__ == "__main__":
    countvec()



"""
运行结果如下：
['我不喜欢python', '我喜欢python', '生命太长', '生命短暂']
[[0 2 0 1]
 [1 0 1 0]]


 当
 data = ["生命 短暂，我 喜欢 python，我 喜欢 python",
        "生命 太长，我 不喜欢 python"]
运行结果如下：
['python', '不喜欢', '喜欢', '太长', '生命', '短暂']
[[2 0 2 0 1 1]
 [1 1 0 1 1 0]]
"""