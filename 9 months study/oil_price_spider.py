import requests
from lxml import etree
import mysql.connector as my

# pip install mysql-connector-python
url='https://www.iamwawa.cn/oilprice.html'
# 创建头部信息
headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.86 Safari/537.36'}
response = requests.get(url = url, headers=headers)
#print(response.text)
html_xpath = etree.HTML(response.text, etree.HTMLParser())
table_xpath = html_xpath.xpath("/html/body/div[2]/div[1]/div[2]/table")[0]
# print(table_xpath)
table_header = table_xpath.xpath("./thead/tr//th/text()")
print(table_header)
table_content = table_xpath.xpath("./tbody//tr//td/text()")
print(table_content)


mydb = my.connect(
    host="localhost",
    user="root",
    password="123456",
    database="oil_price"
)
conf = mydb.cursor()


for i in range(0, len(table_content), 8):
    site = table_content[i]
    oil_98 = table_content[i + 1]
    oil_95 = table_content[i + 2]
    oil_92 = table_content[i + 3]
    oil_0 = table_content[i + 4]
    oil_n_10 = table_content[i + 5]
    oil_n_20 = table_content[i + 6]
    oil_n_35 = table_content[i + 7]
    print(site,oil_98, oil_95, oil_92, oil_0, oil_n_10, oil_n_20, oil_n_35)
    sql = "insert into oil value(%s, %s, %s, %s, %s, %s, %s, %s)"
    data = (site,oil_98, oil_95, oil_92, oil_0, oil_n_10, oil_n_20, oil_n_35)
    conf.execute(sql, data)
mydb.commit()
    # 一般把事务的提交放在循环外面，这一系列插入数据的操作视为一个事务，当要对每个子句执行一次事务提交时才放在循环内部
    # 结束之后要关闭
conf.close()
mydb.close()





