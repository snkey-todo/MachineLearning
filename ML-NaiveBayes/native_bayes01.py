# 导入数据集
from sklearn.datasets import fetch_20newsgroups
# 导入朴素贝叶斯算法
from sklearn.naive_bayes import MultinomialNB
# 导入词频统计
from sklearn.feature_extraction.text import TfidfVectorizer
# 数据集分割
from sklearn.model_selection import train_test_split
# 模型评估
from sklearn.metrics import classification_report

def navie_bayes():
    """
    朴素贝叶斯文本分类
    :return:
    """
    # 加载数据集
    news = fetch_20newsgroups(subset="all")

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 导入tfidf，统计词的重要性
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    #print(x_train.toarray())
    #print(x_test.toarray())
    #print(tf.get_feature_names())

    # 朴素贝叶斯
    navie_bayes = MultinomialNB(alpha=1.0)
    # 训练
    navie_bayes.fit(x_train, y_train)
    # 预测
    y_predict = navie_bayes.predict(x_test)
    print("预测结果：", y_predict)
    # 准确率
    accuracy = navie_bayes.score(x_test, y_test)
    print("准确率:", accuracy)

    # 模型评估
    # target_names是文章类别字符串
    # 精确率、召回率、F1-score、Support(样本数)
    print("模型预测结果：",classification_report(y_test,y_predict,target_names=news.target_names))

if __name__ == "__main__":
    navie_bayes()