# native_bayes01

sklearn20类新闻分类，20个新闻组数据集包含20个主题的18000个新闻组帖子。

算法开发流程如下：

1. 加载20类新闻数据，并进行分割。
    - 该数据集已经集成到`sklearn.dataset`API中，直接使用即可，第一次加载数据会进行数据集的下载。`from sklearn.datasets import fetch_20newsgroups`
    - 将数据集进行分割，分割为训练集和测试集。`from sklearn.model_selection import train_test_split`
2. 生成文章特征词
    - 调用tfidf统计词的重要性。`from sklearn.feature_extraction.text import TfidfVectorizer`
3. 朴素贝叶斯estimator流程进行预估
    - 导入朴素贝叶斯算法。`from sklearn.naive_bayes import MultinomialNB`
    - 训练数据。
    - 计算预测值
    - 计算准确率
