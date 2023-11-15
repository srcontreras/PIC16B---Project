import pandas as pd
import numpy as np
#from tqdm.notebook import tqdm
from yahoo_fin import stock_info as si
from matplotlib import pyplot as plt
from plotly import express as px

starbucks = si.get_data("sbux")
starbucks = starbucks.loc["2021-01-04":] # we want to start in 2021 (ideally)
starbucks["date"] = starbucks.index
starbucks = starbucks.reset_index()
starbucks.drop(columns = "index", inplace = True)

# we will focus on close price

fig = px.line(starbucks, 
              x = "date",
              y = "close",
              hover_data = ["close"],
              title = "Stock price change (real term)")

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
fig.show()

# calculate the percentage change 
# price_tomorrow - price_today / price today

starbucks["tomorrow"] = starbucks["close"].shift(1)
starbucks = starbucks.fillna(starbucks["close"][0])
starbucks["%change"] = (starbucks["tomorrow"] - starbucks["close"]) / starbucks["close"]
avg = np.mean(starbucks["%change"])

fig = px.line(starbucks, 
              x = "date",
              y = "%change",
              hover_data = ["close", "%change"],
              title = "Stock price change (% change)")

fig.add_hline(y = avg,
              line_color = "red")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
fig.show()








