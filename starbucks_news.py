from http import client
from urllib import parse
import json
import pandas as pd

class get_news:
    
    def __init__(self, company_name, end_date, iteration, api_token, conn):
        
        # company_name is the company we focus on
        # end_date is the latest date of news we consider
        # iteration: we cannot retrieve more than 20000 articles at a time
    
        self.company_name = company_name
        self.end_date = end_date
        self.iteration = iteration
        self.api_token = api_token
        self.conn = conn
        
    def retrieve_raw_data(self):
        news_list = []
        
        # we cannot retrieve more than 20000 articles
        # which means we can only have 800 pages, each page 25 articles.
        for i in range(1, self.iteration + 1):
            params = urllib.parse.urlencode({"api_token": self.api_token,
                                             "search": self.company_name,
                                             "language": "en",
                                             "published_before": self.end_date,
                                             "page": i,
                                             "limit": 25})
            self.conn.request("GET", "/v1/news/all?{}".format(params))
            response = self.conn.getresponse()
            data = response.read()
            info = data.decode("utf-8") # information of each news article
            json_info = json.loads(info)
            
            news_list.append(json_info)
        
        return news_list
    
    def element(self, raw_data, objects):
        elements = []
        
        for i in raw_data:
            each_data = i["data"]
            for j in each_data:
                elements.append(j[objects])
        
        return elements 
    

if __name__ == "__main__":
    api_token = "1XOrWumzd3Lw99zn156obKbYmWS9wtVw4FX4LiuS"
    conn = http.client.HTTPSConnection('api.thenewsapi.com')
    elements = ["title", "description", "keywords", 
                "url", "published_at", "source", "categories"]
    df = pd.DataFrame()
    
    starbucks = get_news("Starbucks", "2023-11-09", 800, api_token, conn)
    raw_data = starbucks.retrieve_raw_data()
    
    for x in elements:
        ele = starbucks.element(raw_data, x)
        df[x] = ele
        
    df.to_csv("Starbucks_news")
