class get_news:
    
    def __init__(self, company_name = None, end_date = None, pages = None, 
                 api_token = None, conn = None):
        
        # company_name is the company we focus on
        # end_date is the latest date of news we consider
    
        self.company_name = company_name
        self.end_date = end_date
        self.pages = pages
        self.api_token = api_token
        self.conn = conn
        
    def retrieve_raw_data(self):
        news_list = []
        
        for i in tqdm(range(1, self.pages + 1)):
            params = parse.urlencode({"api_token": self.api_token,
                                             "search": self.company_name,
                                             "search_fields": "description,title,keywords",
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
    
    def clean(self, df):
        df["date"] = df["published_at"].str[:10]
        df["where"] = df["source"].str.split(".").str[:-1].str.join(" ")
        df = df.sort_values(by = "date").reset_index()
        df.drop(columns = ["published_at", "source", "index"], inplace = True)
        
        return df
    
    def understand_data(self, df):
        '''
        to run visualization, do the following:
        object = get_news()
        fig = object.understand_data(data)
        fig
        '''
        yearly_info = df.groupby("date").apply(len).reset_index()
        yearly_info = yearly_info.rename(columns = {0: "news_each_day"})
        yearly_info["year"] = yearly_info["date"].str.split("-").str[0]
        yearly_info["days"] = yearly_info.groupby("year")["year"].transform(len)
        yearly_info["median"] = round(yearly_info.groupby("year")["news_each_day"].transform(np.median))
    
    
        fig = px.scatter(yearly_info,
                         x = "date",
                         y = "news_each_day",
                         color = "year",
                         hover_name = "days",
                         marginal_y = "box",
                         title = "Starbucks",
                         width = 800, 
                         height = 450)
        fig.update_layout(margin={"r":130, "t":30, "l":0, "b":0})
    
        fig.show()
    

if __name__ == "__main__":
    import json
    import pandas as pd
    import numpy as np
    from http import client
    from urllib import parse
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from plotly import express as px
    
#     api_token = "1XOrWumzd3Lw99zn156obKbYmWS9wtVw4FX4LiuS"
#     conn = client.HTTPSConnection('api.thenewsapi.com')
#     elements = ["title", "description", "keywords", 
#                 "url", "published_at", "source", "categories"]
#     df = pd.DataFrame()

#     # use page = 1 to get the total number of news first
#     trial = get_news("Starbucks", "2023-11-15", 1, api_token, conn) 
#     trial_data = trial.retrieve_raw_data()
#     news_number = int(trial_data[0]["meta"]['found'])
#     pages = news_number // 25
#     if news_number % 25 != 0:
#         pages += 1
    
#     starbucks = get_news("Starbucks", "2023-11-15", pages, api_token, conn)
#     starbucks_data = starbucks.retrieve_raw_data()

#     for x in tqdm(elements):
#         ele = starbucks.element(starbucks_data, x)
#         df[x] = ele
    
#     clean_df = starbucks.clean(df)
#     starbucks.understand_data(clean_df)
#     clean_df.to_csv("Starbucks_news.csv")

#   if you want to work on a new company, comment the code below and un-comment the code above.
    
    # if you want to simply visualize the data for starbucks, do:
    df = pd.read_csv("Starbucks_news.csv")
    starbucks = get_news()
    starbucks.understand_data(df)
    
