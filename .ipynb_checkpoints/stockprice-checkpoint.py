class stock_price_visualization:
    def __init__(self, code = None, start = "2021-01-04"):
        '''
        code: company name
        start: start date, note that it has to be valid after 2021
        '''
        
        self.code = code
        self.start = start
        
    def create_data(self):
        company = si.get_data(self.code)
        company = company.loc[self.start:] # 01-04 is the first Monday of 2021.
        company["date"] = company.index
        company = company.reset_index()
        company.drop(columns = "index", inplace = True)
        
        return company
    
    def stock_price_change(self, data):
    
        # we will focus on close price
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
        data["yesterday"] = data["close"].shift(1)
        data = data.fillna(data["close"][0])
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









