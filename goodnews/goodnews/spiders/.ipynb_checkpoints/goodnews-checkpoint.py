# to run 
# scrapy crawl goodnews -o articles.csv

import scrapy

# this scrapy will pull all the headlines and contents of the pages resulting from searching business 
# on the Good News Network
class GoodNews(scrapy.Spider):
    name = 'goodnews' # name of unique spider
    
    # begin with searching up business in goodnewsnetwork website
    start_urls = ['https://www.goodnewsnetwork.org/?s=business']
    
    def parse(self, response):
        """
        Assumes we are starting at goodnewsnetwork website with having searched for business
        Yields parse_title_content for every main article
        """
        
        # not considering iterating through pages
        
        # for every url
        for page in response.css("h3.entry-title.td-module-title > a::attr('href')").extract():
            if page:
                page = response.urljoin(page)
                yield scrapy.Request(page, callback = self.parse_title_content)
            
    # parse_title_content assumes you start at the article page  
    def parse_title_content(self, response):
        """
        Assumes we start at an article page
        Yields dictionary to store article title and content
        """
        # for evey news article
        # record title
        title = response.css("h1.entry-title::text").extract()
        # record content
        content = response.css("div.td-post-content > p::text").extract()
        yield {"title" : title, "content" : content}