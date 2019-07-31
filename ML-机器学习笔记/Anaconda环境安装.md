# 下载

https://www.anaconda.com/


## Mac环境安装(pkg)

下载pkg安装包，直接安装即可。默认自带python3.7环境，并集成了机器学习的常见所有库。

参考：https://blog.csdn.net/depei_yan/article/details/79842243



## Mac环境安装(sh脚本安装)

（1）下载

我选择sh脚本进行安装：https://docs.conda.io/en/latest/miniconda.html

（2）配置默认使用国内镜像加速

```bash
# 配置清华PyPI镜像(如无法运行，将pip版本升级到>=10.0.0)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

（3）创建虚拟环境

首先编写环境依赖软件文件`environment.yml`，内容示例如下：

```bash
name: gluon
dependencies:
- python=3.6
- pip:
  - mxnet==1.4.0
  - d2lzh==0.8.11
  - jupyter==1.0.0
  - matplotlib==2.2.2
  - pandas==0.23.4

```

文件说明：
隔离环境名称为：gluon
python版本：3。6
软件环境：略

执行`conda env create -f environment.yml`创建隔离环境。

（4）激活隔离环境

```bash
conda activate gluon
```

（5）打开jupyter

```bash
jupyter notebook
```

