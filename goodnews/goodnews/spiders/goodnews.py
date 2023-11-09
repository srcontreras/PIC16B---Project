# to run 
# scrapy crawl tmdb_spider -o articles.csv

import scrapy

# this scrapy will pull all the headlines and contents of the pages resulting from searching business 
# on the Good News Network
class GoodNews(scrapy.Spider):
    name = 'goodnews' # name of unique spider
    
    # begin with searching up business in goodnewsnetwork website
    start_urls = ['https://www.goodnewsnetwork.org/?s=business']
    
    # this parse method assumes we are starting at the Awkward. show page
    def parse(self, response):
        
        # not considering iterating through pages
        
        # for every url
        for page in response.css("h3.entry-title.td-module-title > a::atr('href')").extract():
            if page:
                page = response.urljoin(page)
                yield scrapy.Request(page, callback = self.parse_title_content)
            
    # parse_title_content assumes you start at the article page  
    def parse_title_content(self, response):
        # for evey news article
        # record title
        title = response.css("h1.entry-title a::text").get()
        # record content
        content = response.css("div.td-post-content > p:not(strong)").get()
        for project in response.css("td.role.true.account_adult_false.item_adult_false"):
            movie_or_TV_name = project.css("bdi::text").get() # record movie/show name
            # yield dictionary with actor name and project name
            yield {"actor" : actor_name, "movie_or_TV_name" : movie_or_TV_name}