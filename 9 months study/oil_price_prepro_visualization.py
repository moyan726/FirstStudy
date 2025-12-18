import mysql.connector as my
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mydb = my.connect(
    host="localhost",
    user="root",
    password="123456",
    database="oil_price"
)
conf = mydb.cursor()

sql = "select * from oil"

conf.execute(sql)

# 读取全部数据
data_list = conf.fetchall()
print(data_list, type(data_list))
print()
for row_dict in data_list:
    print(row_dict, type(row_dict))

# 关闭
conf.close()
mydb.close()

# data_list 转换为一个 DataFrame， 并指定列名

df = pd.DataFrame(data_list, columns=['site','oil_98', 'oil_95', 'oil_92', 'oil_0', 'oil_n_10', 'oil_n_20', 'oil_n_35'])

# 打印 DataFrame
print(df)

# 查看每一列数据类型

print(df.dtypes)

# 统计 每一列存在 ‘-’ 的数量
# 自定义函数，用于计算一行中所有非NaN单元格中 '-' 符号的总数
def count_dashes_in_row(row):
    # 使用 sum 和 generator expression 来计算非NaN单元格中 '-' 的总数
    return sum(s.count('-') for s in row if pd.notna(s))


# 按行应用自定义函数，并将结果存储在一个新列中
df['dash_count_per_row'] = df.apply(count_dashes_in_row, axis=1)

# 打印新列以查看结果
print(df)

# 替换符号‘-’ 为空
df_replaced = df.replace('-', '')
print(df_replaced)

# 删除 df_replaced 的最后一列
df_dropped = df_replaced.drop('dash_count_per_row', axis=1)
print(df_dropped)

# 将字符型列转换为数值型
# 定义一个函数来转换列，无法转换的值将被设置为 NaN
def convert_to_numeric(series):
    return pd.to_numeric(series, errors='coerce')


# 应用转换函数到每一列，但只针对对象（字符串）类型的列
for col in df_dropped.columns[1:]:
    if df_dropped[col].dtype == 'object':
        df_dropped[col] = convert_to_numeric(df_dropped[col])

# 打印转换后的 DataFrame
print(df_dropped)

# 查看转换后的数据类型
print(df_dropped.dtypes)

# 删除包含 NaN 值的列
oil_data = df_dropped.dropna(axis=1)

# 打印删除后的 DataFrame
print("\n删除包含 NaN 值的列后的 DataFrame:")
print(oil_data)

# 统计分析
# 查询 DataFrame 中的行数
number_of_rows = oil_data.shape[0]
print(f"DataFrame 中的行数: {number_of_rows}")

# 计算油价每一列的最大值
max_values = oil_data.iloc[:, 1:].max()

# 计算油价每一列的最小值
min_values = oil_data.iloc[:, 1:].min()

# 计算油价每一列的平均数
mean_values = oil_data.iloc[:, 1:].mean()

# 计算油价每一列的中位数
median_values = oil_data.iloc[:, 1:].median()

# 输出结果
print("最大值:")
print(max_values)
print("\n最小值:")
print(min_values)
print("\n平均数:")
print(mean_values)
print("\n中位数:")
print(median_values)


# 数据可视化
# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# 设置图形的大小
plt.figure(figsize=(12, 8))

# 绘制三条折线图
# plt.plot(oil_data['site'], oil_data['oil_98'], marker='o', label='98号油价')
plt.plot(oil_data['site'], oil_data['oil_95'], marker='o', label='95号油价')
plt.plot(oil_data['site'], oil_data['oil_92'], marker='o', label='92号油价')

# 设置图表的标题和轴标签
plt.title('不同地方的油价趋势')
plt.xlabel('地方')
plt.ylabel('油价 (元/升)')

# 显示图例
plt.legend()

# 旋转 x 轴上的标签以便更好地显示
plt.xticks(rotation=45)

# 紧凑布局以防止标签重叠
plt.tight_layout()

# 显示图表
plt.show()




