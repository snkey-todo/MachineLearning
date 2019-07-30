# KNN01

## pd.to_datetime()

它是Panda的一个API，可以将`时间戳数据`转换为`多少秒`或者`多少毫秒`。

示例：

```python
time_value=pd.to_datetime(data['time'],unit='s')
```

## pd.DatetimeIndex()

它是Panda的一个API，可以将时间数据转换为字典数据。

示例：

```python
 #把日期格式转换为字典格式
time_value = pd.DatetimeIndex(time_value)

#构造一些特征
day = time_value.day
hour = time_value.hour
weekday = time_value.weekday
```

