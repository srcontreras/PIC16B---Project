import json
import pandas as pd
import numpy as np
from http import client
from urllib import parse
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
    
    Attributes:
    ----------
    company_name: str, name of the company; we want our news article to focus on this company; default Starbucks;
    end_date: str, the date of the latest news article that we incorporate; default 2023-11-14;
    pages: int, the pages of news articles that are about to be retrieved; default None;
    api_token: str, the api key for accessing the TheNewsAPI; default None;
    conn: object, connector that allows us to access the API; default None.
    
    Methods:
    -------
    retrieve_raw_data(): collect all the pages that contain news article from 2021 up to 2023-11-14;
    element(raw_data, objects): store the feature of each news article;
    clean(df): clean the raw dataset;
    understand_data(df): visualize the number of news articles each day.
    '''
    
    def __init__(self, company_name = 'Starbucks', end_date = '2023-11-14', pages = None, 
                 api_token = None, conn = None):
        '''
        @param company_name: str, we only focus on news articles of this company; default Starbucks
        @param end_date: str, we only consider news articles up to this date; default 2023-11-14
        @param pages: int, the number of pages retrieved from the API; default None
        @param api_token: str, the api key for accessing the API; default None
        @param conn: object, the connector that allows to access the API; default None
        '''
        
        self.company_name = company_name
        self.end_date = end_date
        self.pages = pages
        self.api_token = api_token
        self.conn = conn
        
    def retrieve_raw_data(self):
        '''
        This method helps us collect news articles of Starbucks from the TheNewsAPI. We append each page
        from the API to a list, and use it as our raw data for future processes. No parameter is required:
        all necessary ones are defined in the constructor.
        
        @rvalue: list, the return value is a list that contains all pages of news articles of Starbucks. 
        '''
        
        news_list = [] # place to store raw data
        
        # append each page to the list
        for i in range(1, self.pages + 1):
            
            #general settings of accessing the API
            params = parse.urlencode({"api_token": self.api_token,
                                             "search": self.company_name,
                                             "search_fields": "description,title,keywords",
                                             "language": "en",
                                             "published_before": self.end_date,
                                             "page": i,
                                             "limit": 25})
            self.conn.request("GET", "/v1/news/all?{}".format(params)) # connect to the API
            response = self.conn.getresponse()
            data = response.read()
            info = data.decode("utf-8") # information of each page of Starbucks news article
            json_info = json.loads(info) # the info is json format, we transform it into Python dict
            
            news_list.append(json_info)
        
        return news_list
    
    def element(self, raw_data, feature):
        '''
        This method collects details (features) of the news article from the raw data we get from 
        retrieve_raw_data() method. To be specific, we consider 7 features: 
        "title", "description", "keywords", "url", "published_at", "source", and "categories". 
        This method focuses on one of them and find the corresponding feature in the raw data, 
        extract it and save it into a list. To cover all the 7 features, we will use a for loop in the main scope.
        
        @param raw_data: list, the raw data that contains all pages of the news articles;
        @feature: str, one of the 7 features listed above.
        
        @rvalue: list, the return value is a list that contains the corresponding feature of each news article
                 in the raw data.
        '''
        
        elements = [] # place to store the feature of all news articles
        
        # loop over each page
        for i in raw_data:
            each_data = i["data"] # the news articles of one page are stored as value of key 'data'
            for j in each_data: # loop through each news articles
                elements.append(j[feature]) # extract the corresponding feature
        
        return elements 
    
    def clean(self, df):
        '''
        This method cleans the dataframe of Starbucks news articles, such as truncate the publish date of 
        each article, clean the source, and sort the dataframe based on publish date, from oldest to latest.
        
        @param df: dataframe, each column of the dataframe is one feature of the news article, which is returned
                   by the element(raw_data, feature) method. 
        
        @rvalue: dataframe, which is the cleaned version of the original df.
        '''
        
        df["date"] = df["published_at"].str[:10] # Year-Month-Date
        df["where"] = df["source"].str.split(".").str[:-1].str.join(" ") # clean source
        df = df.sort_values(by = "date").reset_index() # sort from oldest to latest news articles
        df.drop(columns = ["published_at", "source", "index"], inplace = True)
        
        return df
    
    def understand_data(self, df):
        '''
        This method visualizes the number of news articles on each day, along with the median of number of each year.
        This can help understand the distribution of our data, which can be used to make inference of the 
        performance of prediction later. 
        
        @param df: dataframe, the cleaned version of dataset, returned by clean(df).
        
        No return value, but show the plot.
        '''
        
        yearly_info = df.groupby("date").apply(len).reset_index() # number of news on each day
        yearly_info = yearly_info.rename(columns = {0: "news_each_day"})
        yearly_info["year"] = yearly_info["date"].str.split("-").str[0] # get year
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

#***************************************************************************#
    
# we've already retrieved and saved the news article; it's recommended to read the .csv file directly:
    df = pd.read_csv("Starbucks_news.csv")
    starbucks = get_news()
    starbucks.understand_data(df)
    
