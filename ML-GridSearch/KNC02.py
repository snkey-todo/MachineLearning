# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd


def knncls():
    """
    K-近邻预测用户签到位置
    :return:None
    """
    # 读取数据
    data = pd.read_csv("/Users/zhusheng/WorkSpace/Dataset/1-facebook/facebook-v-predicting-check-ins/train.csv")

    # print(data.head(10))

    #处理数据
    #1.缩小数据,查询数据筛选
    data=data.query("x>1.0&x<1.25&y>2.5&y<2.75")

    #处理时间的数据
    time_value=pd.to_datetime(data['time'],unit='s')

    # print(time_value)

    #把日期格式转换为字典格式
    time_value=pd.DatetimeIndex(time_value)

    #构造一些特征
    data['day']=time_value.day
    data['hour']=time_value.hour
    data['weekday']=time_value.weekday

    #把时间戳特征删除
    data=data.drop(['time'],axis=1)

    # print(data)

    # 把签到数量少于n个目标位置删除
    place_count = data.groupby('place_id').count()

    tf = place_count[place_count.row_id > 3].reset_index()

    data = data[data['place_id'].isin(tf.place_id)]

    # 取出数据当中的特征值和目标值
    y = data['place_id']

    x = data.drop(['place_id'], axis=1)
    x = data.drop(['row_id'], axis=1)

    # 进行数据的分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


    #特征工程（标准化）
    std=StandardScaler()

    #对测试集和训练集的特征值进行标准化
    x_train=std.fit_transform(x_train)

    x_test=std.transform(x_test)

    # 进行算法流程
    # 使用网格搜索调优超参数的时候，创建算法的时候就不需要指定超参数，否则网格搜索的超参数不生效
    knn = KNeighborsClassifier()

    # 进行网格搜素取了3个超参数，每个超参数都进行十折交叉验证
    gc = GridSearchCV(knn, param_grid={"n_neighbors": [3,5,10]}, cv=10)

    gc.fit(x_train,y_train)

    # 预测准确率
    print("预测结果：", gc.predict(x_test))
    print("预测准确率：", gc.score(x_test, y_test))
    print("在交叉验证中最好大的结果：", gc.best_score_)
    print("选择最好的模型是:", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果：", gc.cv_results_)

    return None

if __name__ == "__main__":
    knncls()


"""
运行结果如下：
预测结果： [5270522918 1435128522 2355236719 ... 3683087833 3312463746 6683426742]
预测准确率： 0.6936170212765957
在交叉验证中最好大的结果： 0.6788303909205549
选择最好的模型是: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform')
每个超参数每次交叉验证的结果： {'mean_fit_time': array([0.00705981, 0.00625448, 0.00631008]), 'std_fit_time': 
array([0.00105088, 0.00030914, 0.00040351]), 'mean_score_time': 
array([0.05117078, 0.05543509, 0.06594858]), 'std_score_time': 
array([0.00453885, 0.00610629, 0.00765468]), 'param_n_neighbors': masked_array(data=[3, 5, 10],
             mask=[False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 10}], 
            'split0_test_score': array([0.62944162, 0.63306744, 0.61783901]), 
            'split1_test_score': array([0.64718773, 0.64207451, 0.6274653 ]), 
            'split2_test_score': array([0.65998515, 0.66072754, 0.64587973]), 
            'split3_test_score': array([0.67709924, 0.6740458 , 0.65801527]), 
            'split4_test_score': array([0.66068643, 0.65600624, 0.65522621]), 
            'split5_test_score': array([0.69230769, 0.69871795, 0.67227564]), 
            'split6_test_score': array([0.69622332, 0.70114943, 0.68555008]), 
            'split7_test_score': array([0.68838764, 0.69172932, 0.68838764]), 
            'split8_test_score': array([0.7158609 , 0.7158609 , 0.69974555]), 
            'split9_test_score': array([0.72735116, 0.73252804, 0.70664366]), 
            'mean_test_score': array([0.6778058 , 0.67883039, 0.66401324]), 
            'std_test_score': array([0.02900632, 0.03092579, 0.02853444]), 
            'rank_test_score': array([2, 1, 3], dtype=int32), 
            'split0_train_score': array([0.8087364 , 0.76938721, 0.72322929]), 
            'split1_train_score': array([0.80475307, 0.76985599, 0.71755455]), 
            'split2_train_score': array([0.80460277, 0.76642271, 0.71492814]), 
            'split3_train_score': array([0.80383196, 0.76560028, 0.71436105]), 
            'split4_train_score': array([0.80291075, 0.76494827, 0.71637735]), 
            'split5_train_score': array([0.80253497, 0.76730769, 0.71293706]), 
            'split6_train_score': array([0.8016565 , 0.7646905 , 0.71150828]), 
            'split7_train_score': array([0.80323732, 0.76555565, 0.71290575]), 
            'split8_train_score': array([0.80189417, 0.76348944, 0.71413676]), 
            'split9_train_score': array([0.80119698, 0.76294562, 0.71107642]), 
            'mean_train_score': array([0.80353549, 0.76602034, 0.71490146]), 
            'std_train_score': array([0.00207201, 0.0021688 , 0.00336799])}
"""