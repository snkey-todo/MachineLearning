from sklearn.decomposition import PCA

def pca():
    """
    使用主成分分析进行特征降维
    :return:
    """
    # 参数说明：
    # n_components为超参数，也就是说降维后的数据保持降维前数据90%的特征
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
    print(data)

if __name__ == "__main__":
    pca()

"""
运行结果如下：
[[ 1.22879107e-15  3.82970843e+00]
 [ 5.74456265e+00 -1.91485422e+00]
 [-5.74456265e+00 -1.91485422e+00]]
"""