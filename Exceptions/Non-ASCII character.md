# 错误信息

File "TF4-CNN/mnist.py", line 7
SyntaxError: Non-ASCII character '\xe6' in file TF4-CNN/mnist.py on line 7, but no encoding declared; see http://python.org/dev/peps/pep-0263/ for details

## 解决办法

文件行首增加如下代码

```python
# -*- coding: utf-8 -*-
```
