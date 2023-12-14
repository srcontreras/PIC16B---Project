import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
from matplotlib import pyplot as plt
from plotly import express as px

class stockprice:
    '''
    This class is for interacting with the dataset of stock price of Starbucks, including collecting the data,
    visualizing the data, and adding new features to the new data. The data is the general stock price of 
    Starbucks ranging from 2021-01-04 to 2023-11-14, including volume, open, close, high, low and so on. 
    Package yahoo_fin is used.
    
    Attributes:
    ----------
    code: str, the ticker symbol of a company; default is sbux, stands for Starbucks;
    start: str, start date of the stock price; default is 2021-01-04;
    end: str, end date of the stock price; default is 2023-11-14.
    
    Methods:
    -------
    create_data(): create the stock price dataset with the given restriction;
    stock_price_change(data): visualize the daily price change of stock price;
    stock_price_percentage(data): visualize the daily percentage price change of stock price;
    (some others could be added)
    '''
    
    def __init__(self, code = 'sbux', start = "2021-01-04", end = "2023-11-14"):
        '''
        @param code: str, ticker symbol of company, default is sbux;
        @param start: str, start date of the stock price, default is 2021-01-03;
        @param end: str, end date of the stock price, default is 2023-11-14
        '''
        
        self.code = code
        self.start = start
        self.end = end
        
    def create_data(self):
        '''
        This method creates the stock price dataset with code, start, end specified in the constructor function.
        No input parameters are required. We use a package called yahoo_fin to retrieve data. 
        
        @rvalue: dataframe, the return value is the pandas dataframe that contains the desired stock price. 
        '''
        
        company = si.get_data(self.code)
        company = company.loc[self.start:self.end] 
        company["date"] = company.index # create a data column
        company = company.reset_index() # reset_index so that index starts from 0
        company.drop(columns = "index", inplace = True)
        
        return company
    
    def stock_price_change(self, data):
        '''
        This method creates a plot that visualizes the daily stock changes in real terms. Close price of Starbucks
        is considered, but not open price, as the former is more representative.
        
        @param data: dataframe, the dataset we get from method create_data()
        
        The method has no return values, but plot a figure to present daily stock changes.
        '''

        fig = px.line(data, 
                      x = "date",
                      y = "close",
                      hover_data = ["close"],
                      title = "Stock price change (real term)")

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        fig.show()
        
    def stock_price_percentage(self, data):
        '''
        This method creates a plot that visualizes the daily stock changes in percentage terms. 
        Again, close price of Starbucks is considered.
        
        @param data: dataframe, the dataset we get from method create_data()
        
        The method has no return values, but plot a figure to present daily percentage stock changes.
        '''
        
        # percentage change satisfies (today - yesterday) / yesterday
        data["yesterday"] = data["close"].shift(1) # shift the column downward by 1 row
        data = data.fillna(data["close"][0]) # fill nan with the first entry of close
        data["%change"] = (data["close"]- data["yesterday"]) / data["yesterday"]
        avg = np.mean(data["%change"])

        fig = px.line(data, 
                      x = "date",
                      y = "%change",
                      hover_data = ["close", "%change"],
                      title = "Stock price change (% change)")

        fig.add_hline(y = avg,
                      line_color = "red",
                      annotation_text = "{}%".format(round(avg*100, 3)),
                      annotation_position="bottom left",
                      annotation_font_size=25,
                      annotation_font_color="black")

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        fig.show()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from yahoo_fin import stock_info as si
    from matplotlib import pyplot as plt
    from plotly import express as px
    
    stock = stock_price_visualization("sbux")
    data = stock.create_data()
    figure1 = stock.stock_price_change(data)
    figure2 = stock.stock_price_percentage(data)









