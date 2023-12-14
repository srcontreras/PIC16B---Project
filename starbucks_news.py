import json
import pandas as pd
import numpy as np
from http import client
from urllib import parse
from tqdm import tqdm
from matplotlib import pyplot as plt
from plotly import express as px

class get_news:
    '''
    ***** IMPORTANT NOTE *****
    The News API used is not free. Anyone who runs our project shouldn't run the methods in this class,
    except understand_data(df), which presents plots. Instead, we have created Starbucks_news.csv dataset, 
    which contains all the textual information we need. 
    Direcly reading from this .csv dataframe is recommended. 
    **************************
    
    This class is for retrieving the news article of Starbucks from 2021-01-01 to 2023-11-14. A news API called
    TheNewsAPI is used for data collection. Data cleaning and visualization are also implemented in this class.
    As notified, the API is not free, so re-running is not recommended. To see how to retrieve data from this 
    API, see the code in __main__() at the bottom.

    '''
    def __init__(self, company_name = 'Starbucks', end_date = '2023-11-14', pages = None, 
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
        
        for i in range(1, self.pages + 1):
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
    
    # settings of the TheNewsAPI
    api_token = "1XOrWumzd3Lw99zn156obKbYmWS9wtVw4FX4LiuS"
    conn = client.HTTPSConnection('api.thenewsapi.com')
    elements = ["title", "description", "keywords", 
                "url", "published_at", "source", "categories"] # columns of the dataset
    df = pd.DataFrame()

    # use page = 1 to get the total number of news first
    trial = get_news("Starbucks", "2023-11-15", 1, api_token, conn) # initialize the class object
    trial_data = trial.retrieve_raw_data() 
    news_number = int(trial_data[0]["meta"]['found']) # number of news articles
    
    # Each page contains 25 news articles, we want to calculate the total pages we have
    # so that we can loop through the number of pages.
    pages = news_number // 25 
    if news_number % 25 != 0:
        pages += 1
    
    # now we know the number of pages, we call another class object with this number
    starbucks = get_news("Starbucks", "2023-11-15", pages, api_token, conn)
    starbucks_data = starbucks.retrieve_raw_data()
    
    # create df columns, each column contains values of corresponding feature of news articles
    # for example, the column named title contains titles of news articles. 
    for x in elements:
        ele = starbucks.element(starbucks_data, x)
        df[x] = ele
    
    clean_df = starbucks.clean(df) # clean the dataframe
    clean_df.to_csv("Starbucks_news.csv") # export to this .csv file.


    
# if you want to simply visualize the data for starbucks, do:
    df = pd.read_csv("Starbucks_news.csv")
    starbucks = get_news()
    starbucks.understand_data(df)
    
