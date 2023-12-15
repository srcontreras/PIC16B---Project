
import scrapy

# this scrapy will pull all the headlines and contents of the pages resulting from searching business 
# on the Good News Network
class GoodNews(scrapy.Spider):
    '''
    This class helps us web scrape the headlines and content of each news article on Good News Network.
    It's a standard format of how we interact with the scrapy module. Make sure you correctly installed scrapy
    before running this in terminal.
    
    Methods:
    -------
    parse(response): parse the title and content on one page by the parse_title_content(response) method;
                     if there is a 'next page', the code will lead us to that page after finishing parsing on
                     the current page.
    parse_title_content(response): parse the title and content on one page.
    '''
    name = 'goodnews' # name of unique spider, very important
    
    # begin with searching up business in goodnewsnetwork website
    start_urls = ['https://www.goodnewsnetwork.org/?s=business']
    
    def parse(self, response):
        """
        Assumes we are starting at goodnewsnetwork website with having searched for business
        Yields parse_title_content for every main article.
        
        @param response: object, scrapy object that allows us to interact with and parse the website.
        
        @rvalue: no return value, but call the function defined below and do it repetitively until no next page.
        """
        
        # not considering iterating through pages
        
        # for every url
        for page in response.css("h3.entry-title.td-module-title > a::attr('href')").extract():
            if page:
                page = response.urljoin(page)
                yield scrapy.Request(page, callback = self.parse_title_content)
                
        # to retrieve the next page, we can find the link for it and yield back.
        # we do this outside the for-loop
        
        next_page = response.css("div.page-nav.td-pb-padding-side > span.current + a::attr(href)").get()
        if next_page:
            yield scrapy.Request(next_page, callback = self.parse)
            
    # parse_title_content assumes you start at the article page  
    def parse_title_content(self, response):
        """
        Assumes we start at an article page
        Yields dictionary to store article title and content.
        
        @param response: object, scrapy object that allows us to interact with and parse the website.
        
        @rvalue: no return value, but yield a dictionary which includes title and content of a news article.
        """
        # for evey news article
        # record title
        title = response.css("h1.entry-title::text").extract()
        # record content
        content = response.css("div.td-post-content > p::text").extract()
        if content:
            yield {"title" : title, "content" : content}
