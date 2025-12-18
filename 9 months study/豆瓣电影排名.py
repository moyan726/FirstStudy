# https://movie.douban.com/top250?start=25
# 爬取豆瓣电影排行榜start=25 确定开始的影片位置
import requests
from bs4 import BeautifulSoup
#
# class Douban:
#
#     def __init__(self):
#         self.URL = 'https://movie.douban.com/top250'
#         self.star_num = []
#         #循环这个数组从0开始到250结束  步长为25
#         for start_num in range(0, 250,25):
#             self.star_num.append(start_num)
#         #反爬 获取请求头(请求头的实质是键值对 需要用{}括起来)
#         self.header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/11 SLBVPV/64-bit'}
#
#     def get_top250(self):
#         for start in self.star_num:
#             start = str(start)
#             html = requests.get(self.URL, params={'start':start})
#             soup = BeautifulSoup(html.text,'html.parser')
#             #调用beautifulsoup方法中的soup.select()抓取数据
#             name = soup.select('#content > div > div.article > ol > li')
#             print(name)
#
#
#
#
#
#
#
#
#
# if __name__ == '__main__':
#     cls = Douban()
#     cls.get_top250()







import requests
from bs4 import BeautifulSoup

class Douban:
    def __init__(self):
        self.URL = 'https://movie.douban.com/top250'
        self.start_num = []
        # 循环这个数组从0开始到250结束  步长为25
        for start_num in range(0, 250, 25):
            self.start_num.append(start_num)
        # 反爬 获取请求头(请求头的实质是键值对 需要用{}括起来)
        self.header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/11 SLBVPV/64-bit'
        }

    def get_top250(self):
        for start in self.start_num:
            try:
                html = requests.get(self.URL, params={'start': start}, headers=self.header)
                html.raise_for_status()
                soup = BeautifulSoup(html.text, 'html.parser')
                # 选择所有电影项
                movies = soup.select('#content > div > div.article > ol > li')
                for movie in movies:
                    name = movie.select_one('div > div.info > div.hd > a > span.title')
                    if name:
                        print(name.text)
            except requests.RequestException as e:
                print(f"请求出错: {e}")


if __name__ == '__main__':
    cls = Douban()
    cls.get_top250()