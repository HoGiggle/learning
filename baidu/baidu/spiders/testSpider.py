# coding:utf-8
import scrapy
from scrapy.spiders import BaseSpider
from bs4 import BeautifulSoup

class testSpider(scrapy.Spider):
    name = 'test'
    start_urls = ['https://s.2.taobao.com/list/list.htm?spm=2007.1000337.0.0.RrSiSG&catid=50100398&st_trust=1&page=1&ist=0']

    def parse(self, response):

        # //*[contains(concat( " ", @class, " " ), concat( " ", "item-title", " " ))]//a
        for title in response.xpath('//*[@id="J_ItemListsContainer"]/ul/li[*]/div[1]/h4/a'):
            print "****************", title
            # yield scrapy.Request('http:'+title.xpath('@href').extract()[0], self.detail)
            scrapy.Request('http:'+title.xpath('@href').extract()[0], self.detail)

    def detail(self, response):
        data = BeautifulSoup(response)
        print data.title

        # data = BeautifulSoup(response.body, 'lxml')
        # area = data.select('#J_Property > h1')
        # print area[0].get_text()
