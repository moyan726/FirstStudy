Pandas

Jupyter Notebook是什么
Jupyter Notebook是一个开源的web应用程序,一个交互式笔记本，支持运行 40 多种编程语言。
它允许您创建和共享文档,包含代码,方程,可视化和叙事文本。
用途包括:数据清洗和转换,数值模拟,统计建模、数据可视化、机器学习等等。
支持以网页的形式分享，GitHub 中天然支持 Notebook 展示，也可以通过 nbviewer 分享你的文档。当然也支持导出成 HTML、Markdown 、PDF 等多种格式的文档。
不仅可以输出图片、视频、数学公式，甚至可以呈现一些互动的可视化内容，比如可以缩放的地图或者是可以旋转的三维模型。

通过 pip 安装：pip install jupyter notebook
安装成功提示有：jupyter、jupyter-client、jupyter-console、jupyter-core。

打开打开cmd命令提示符窗口输入jupyter notebook 回车，然后浏览器就会打开Jupyter notebook。


import pandas as pd

student = pd.read_table('C:\\Users\\guojl\\Desktop\\students.txt', sep = ',')

print(student)

student[student['age']>23]

student[student['age']>22]

student.groupby('class')['id'].count().reset_index().rename(columns={'id':'count'})

student.groupby('class')['age'].min().reset_index().rename(columns = {'age':'min_age'})

score = pd.read_table('C:\\Users\\guojl\\Desktop\\score.txt', sep = ',')

print(score)

score.columns

score.groupby("id")["score"].sum().reset_index().rename(columns={'score':'sum_score'}).sort_values('sum_score', ascending = False).head()

student.merge(score, on = 'id')



merge(left, right, how=‘inner’, on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=(’_x’, ‘_y’), copy=True, indicator=False, validate=None)

left	参与合并的左侧DataFrame
right	参与合并的右侧DataFrame
how	连接方式：‘inner’（默认）、‘outer’、‘left’、‘right’,分别对应内连接、外连接、左连接、右连接；外连接其实左连接和右连接的并集。左连接是左侧DataFrame取全部数据，右侧DataFrame匹配左侧DataFrame。（右连接right和左连接类似）
on	用于连接的列名，必须同时存在于左右两个DataFrame对象中，如果未指定，则以left和right列名的交集作为连接键
left_on	左侧DataFarme中用作连接键的列
right_on	右侧DataFarme中用作连接键的列
left_index	将左侧的行索引用作其连接键
right_index	将右侧的行索引用作其连接键
sort	根据连接键对合并后的数据进行排序，默认为True。有时在处理大数据集时，禁用该选项可获得更好的性能
suffixes	字符串值元组，用于追加到重叠列名的末尾，默认为（‘_x’,‘_y’）.例如，左右两个DataFrame对象都有‘data’，则结果中就会出现‘data_x’，‘data_y’
copy	设置为False，可以在某些特殊情况下避免将数据复制到结果数据结构中。默认总是赋值

作用：
连接两个DataFrame并返回连接之后的DataFrame

import matplotlib.pyplot as plt
df = student.groupby('class')['id'].count().reset_index().rename(columns={'id':'count'})
plt.rcParams['font.family']=['SimHei']
fig,ax = plt.subplots()
ax.plot(df['class'], df['count'])
ax.set_xticklabels(labels=df['class'], rotation=90)  # 旋转90度
plt.show()


