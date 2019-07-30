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
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    #print(x_train.toarray())

    x_test = tf.transform(x_test)
    #print(x_test.toarray())

    #print(tf.get_feature_names())

    navie_bayes = MultinomialNB(alpha=1.0)
    navie_bayes.fit(x_train, y_train)

    y_predict = navie_bayes.predict(x_test)
    print("预测结果：", y_predict)

    accuracy = navie_bayes.score(x_test, y_test)
    print("准确率:", accuracy)

    # 模型评估
    # target_names是文章类别字符串
    # 精确率、召回率、F1-score、Support(样本数)
    print("模型预测结果：",classification_report(y_test,y_predict,target_names=news.target_names))

if __name__ == "__main__":
    navie_bayes()


"""
运行结果：
预测结果： [14  8 18 ... 16  2 12]
准确率: 0.8548387096774194
模型预测结果：                           precision    recall  f1-score   support

             alt.atheism       0.89      0.74      0.81       198
           comp.graphics       0.94      0.70      0.80       268
 comp.os.ms-windows.misc       0.82      0.85      0.84       241
comp.sys.ibm.pc.hardware       0.67      0.89      0.76       218
   comp.sys.mac.hardware       0.92      0.83      0.87       253
          comp.windows.x       0.95      0.86      0.90       246
            misc.forsale       0.95      0.71      0.81       268
               rec.autos       0.91      0.95      0.93       238
         rec.motorcycles       0.90      0.96      0.93       227
      rec.sport.baseball       0.94      0.96      0.95       255
        rec.sport.hockey       0.93      0.98      0.96       248
               sci.crypt       0.72      0.96      0.83       246
         sci.electronics       0.91      0.80      0.85       245
                 sci.med       0.97      0.92      0.94       251
               sci.space       0.92      0.95      0.93       263
  soc.religion.christian       0.63      0.97      0.76       268
      talk.politics.guns       0.77      0.96      0.86       228
   talk.politics.mideast       0.86      0.99      0.92       211
      talk.politics.misc       0.98      0.64      0.78       185
      talk.religion.misc       1.00      0.21      0.35       155

                accuracy                           0.85      4712
               macro avg       0.88      0.84      0.84      4712
            weighted avg       0.88      0.85      0.85      4712

Support为划分为这个类型的文章的样本数量。
"""