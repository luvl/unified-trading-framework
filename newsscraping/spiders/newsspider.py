import scrapy
import os

import asyncio
from aiohttp import ClientSession, TCPConnector

if os.path.exists('vinamilk.json'):
    os.remove('vinamilk.json')

async def fetch(url):
    async with ClientSession(connector=TCPConnector(verify_ssl=False)) as session:
        async with session.get(url, allow_redirects=True, headers={'User-Agent': 'python-requests/2.20.0'}) as response:
            return response.url

class NewsCafefSpider(scrapy.Spider):
    name = "cafef-vnm-news"
    start_urls = ['http://s.cafef.vn/Ajax/Events_RelatedNews_New.aspx?symbol=VNM&floorID=0&configID=0&PageIndex=1&PageSize=10000&Type=2']

    def parse(self, response):
        self.logger.info('vinamilk-news spider')
        domain = "https://s.cafef.vn/"
        news_list = response.css('div#divEvents li')

        for news in news_list:
            url = domain+news.css('.docnhanhTitle::attr(href)').get()[1:]
            yield {
                'datatime': news.css('.timeTitle::text').get(),
                'url': str(url),
                'title': news.css('.docnhanhTitle::text').get()
            }


class NewsVietStockSpider(scrapy.Spider):
    name = "vietstock-vnm-news"
    start_urls = ['https://finance.vietstock.vn/VNM/tin-moi-nhat.htm']

    def parse(self, response):
        self.logger.info('vinamilk-news spider')

        news_list = response.css('div.col-sm-24.col-md-14 tbody tr')

        for news in news_list:
            yield {
                'datatime': news.css('td.col-date::text').get(),
                'url': news.css('.text-link.news-link::attr(href)').get(),
                'title': news.css('.text-link.news-link::text').get()
            }

class NewsGoogleSpider(scrapy.Spider):
    name = "google-vnm-news"
    start_urls = ['https://news.google.com/topics/CAAqJAgKIh5DQkFTRUFvS0wyMHZNREp4TVRReE1SSUNaVzRvQUFQAQ?hl=en-US&gl=US&ceid=US%3Aen']
    
    # NiLAwe y6IFtc R7GTQ keNKEd: news box
    # WW6dff uQIVzc Sksgp: datatime
    # DY5T1d: title

    def parse(self, response):
        domain = "https://news.google.com"
        self.logger.info('vinamilk-news spider')

        news_list = response.css('div.NiLAwe.y6IFtc.R7GTQ.keNKEd')
        for news in news_list:
            redirect_url = domain+news.css('div.NiLAwe.y6IFtc.R7GTQ.keNKEd a::attr(href)').get()[1:]
            url = asyncio.get_event_loop().run_until_complete(fetch(redirect_url))

            yield {
                'datatime': news.css('.WW6dff.uQIVzc.Sksgp::text').get(),
                'url': str(url),
                'title': news.css('.DY5T1d::text').get()
            }