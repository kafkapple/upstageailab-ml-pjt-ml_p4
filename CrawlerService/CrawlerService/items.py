# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class NaverBlogCrawlerItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    author_name = scrapy.Field()
    date = scrapy.Field()
    preview = scrapy.Field()
    content = scrapy.Field()
    tags = scrapy.Field()    
