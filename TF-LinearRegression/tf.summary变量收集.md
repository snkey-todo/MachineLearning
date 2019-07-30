# tf.summary变量收集

tf.summary变量收集的目的是：观察模型的参数、损失值等变量值的变化。

## 收集变量

`tf.summary.scalar(name=’’,tensor)` ：收集对于损失函数和准确率等单值变量,name为变量的名字，tensor为值。
`tf.summary.histogram(name=‘’,tensor)` ：收集高维度的变量参数。
`tf.summary.image(name=‘’,tensor)` ：收集输入的图片张量能显示图片。

## 合并变量并写入事件文件

merged = tf.summary.merge_all()：合并变量。
summary = sess.run(merged)：运行合并，每次迭代都需运行。
FileWriter.add_summary(summary,i)：写入事件文件，i表示第几次的值。

代码示例：参考`linear_regression_with_scope2.py`
