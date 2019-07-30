from sklearn.feature_extraction.text import CountVectorizer

def countvec():
    """
    对文本进行特征值化
    :return:
    """
    countvec = CountVectorizer()
    data = ["life is short,i like python, i like python",
            "life is too long,i dislike python"]

    result = countvec.fit_transform(data)

    print(countvec.get_feature_names())
    print(result.toarray())

if __name__ == "__main__":
    countvec()



"""
运行结果如下：
['dislike', 'is', 'life', 'like', 'long', 'python', 'short', 'too']
[[0 1 1 2 0 2 1 0]
 [1 1 1 0 1 1 0 1]]

出现了feature_names，次数+1
"""