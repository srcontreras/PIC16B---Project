import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
from matplotlib import pyplot as plt
from plotly import express as px
import plotly.subplots as sp
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split # splitting data
from sklearn.metrics import mean_absolute_error, mean_squared_error  # assessing data
from tensorflow.keras.models import Sequential # type of model
from tensorflow.keras.layers import LSTM, Dense, Dropout # layers
from tensorflow.keras.callbacks import EarlyStopping # model modifications
from sklearn.preprocessing import MinMaxScaler # data manipulation


        
def create_data(code = "SBUX", start = "2021-01-04", end = "2023-11-14"):
    '''
    This function creates the stock price dataset with specified code, start, and end.
    We use a package called yahoo_fin to retrieve data. 
    
    @param code: str, ticker symbol of company, default is sbux;
    @param start: str, start date of the stock price, default is 2021-01-03;
    @param end: str, end date of the stock price, default is 2023-11-14.
    
    @rvalue: dataframe, the return value is the pandas dataframe that contains the desired stock price. 
    '''
    
    company = si.get_data(self.code)
    company = company.loc[self.start:self.end] 
    company["date"] = company.index # create a data column
    company = company.reset_index() # reset_index so that index starts from 0
    company.drop(columns = "index", inplace = True)
        
    return company
    
def stock_price_change(data):
    '''
    This function creates a plot that visualizes the daily stock changes in real terms. Close price of Starbucks
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
        
def stock_price_percentage(data):
    '''
    This function creates a plot that visualizes the daily stock changes in percentage terms. 
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

def clean_data(data):
    '''
    This function drops any NA values from the data.

    @param data: dataframe with stock price data

    The function returns data without NA values.
    '''
    return data.dropna(ignore_index = True)

def add_features(data):
    '''
    This function adds predictor variables to the stock price dataset. The "quarterly_reports" column identifies months when quarterly
    earnings for a public company are released. It adds columns for the difference of the open and closing stock prices, and the low and high
    stock prices. It also adds a column for the estimaged moving average with data from 100 days. Lastly it adds two columns for the Bollinger
    Bands using the moving average and moving average standard deviation to create bounds for the stock prices. The function also sets the 
    date as the index and removes NA values.

    @param data: dataframe with stock price data

    The function returns data with new columns and without NA values.
    '''
    # add column for months when quarter earnings reports are released: 
    split_date = data["date"].astype(str).str.split("-", expand = True) # year, month, day
    data["month"] = split_date[1].astype("int")
    data["day"] = split_date[2].astype("int")
    data["year"] = split_date[0].astype("int")
    data['date'] = pd.to_datetime(data["date"])
    data["quarterly_reports"] = data['date'].dt.month.isin([1, 4, 7, 10]).astype(int)
    
    # add open-close and low-high columns
    data["open-close"] = data["open"] - data["close"]
    data["low-high"] = data["low"] - data["high"]
    
    # add Estimated Moving Average using time window of 100 days
    data['ema100'] = data['close'].ewm(span=100, adjust=False).mean()
    
    # add Bollinger Bands
    data['moving_avg20'] = data['close'].rolling(window=20).mean() # moving average for 20 day period
    data['moving_avg20std'] = data['close'].rolling(window=20).std() # moving average standard deviation for 20 day period
    data['bollinger_upper'] = data['moving_avg20'] + (data['moving_avg20std'] * 2)
    data['bollinger_lower'] = data['moving_avg20'] - (data['moving_avg20std'] * 2)
    
    # set date as index
    data.set_index('date', inplace=True)
    
    # replace NA in "close" column with last valid closing price
    na_index = data.at[data.index.max(), 'close']    
    data.fillna((na_index), inplace=True)
    
    return data

def pred_target(features, data):
    '''
    This function creates two datasets (predictor and target datasets) for the model. 

    @param features: list of column names in dataset that will be used as the predictor data
    @param data: stock price data

    The function returns the dataset filtered by the features and a dataset with just closing prices.
    '''
    return data[features], data["close"]

def scalers(pred, target):
    '''
    This function defines the MinMaxScaler for the predictor and target data, as well as scales the predictor and target data to values from 
    0 to 1.

    @param pred: predictor data
    @param target: target data

    The function returns two scalers and the scaled predictor and target data.
    '''
    # normalize between 0 and 1
    # define scalers
    scaler_pred = MinMaxScaler() # for scaling multiple columns
    scaler_target = MinMaxScaler() # for scaling one column
    # normalize data
    scaled_pred = scaler_pred.fit_transform(pred.values)
    scaled_target = scaler_target.fit_transform(target.values.reshape(-1,1))
    return scaler_pred, scaler_target, scaled_pred, scaled_target

def train_val_test(pred, seq_len, scaled_p):
    '''
    This function defines the training, validation, and testing data. 

    @param pred: predictor data
    @param seq_len (int): represents the number of time steps model will use to predict stock price
    @param scaled_p: scaled predictor data

    The function returns the training, validation, and test data.
    '''
    train_len = int(0.7*len(pred)) # define training length
    val_len = int(0.2*train_len) # validations set is 20% of training set
    
    val_data = scaled_p[:val_len,:] # subset validation data
    train_data = scaled_p[val_len:train_len, :] # subset training data
    test_data = scaled_p[train_len - seq_len:, :] # subset test data with seq_len
    
    return train_data, val_data, test_data

def split_data(seq_len, data, index_close):
    '''
    This function splits time series data into input sequences and corresponding output values for single-step prediction.

    @param seq_len (int): represents the number of time steps model will use to predict stock price
    @param data: 2D array containing time series data where rows represent time steps and columns represent features
    @index_close (int): index of the column containing the target variable (closing prices)

    The function returns:
    x: input sequences each of shape (seq_len, num_features)
    y: corresponding output values, representing the target variable for single-step prediction
    '''
    x, y = [], []
    data_len = data.shape[0]

    for i in range(seq_len, data_len):
        x.append(data[i-seq_len:i,:]) # previous 50 day values
        y.append(data[i, index_close]) #contains the prediction values for validation, for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

def compile_train(model, x_train, y_train, x_val, y_val):
    '''
    This function takes in a model and compiles it using the mean squared error as the loss function. It then trains the model on the training
    and validation data using callbacks.

    @param model: Sequential model to predict stock price
    @param x_train: training input sequence data
    @param y_train: training target data
    @param x_val: validation input sequence data
    @param y_val: validation target data

    The function returns the history and metrics of the model as it is being trained. 
    '''
    # compile the model 
    model.compile(optimizer="adam", loss="mean_squared_error")

    # train Model
    callback = EarlyStopping(monitor='loss', patience=5, verbose=1)
    return model.fit(x_train, y_train, 
                    batch_size = 16, 
                    epochs = 50,
                    callbacks = [callback],
                    validation_data=(x_val, y_val)
                   )

def plot_history(history):
    '''
    This function plots the model loss for each epoch of the model's performance on the training and validation data.

    @param history: History object of a model

    The function does not return anything and instead plots the model loss. 
    '''
    # create subplot 
    fig = sp.make_subplots(rows=1, cols=1, subplot_titles=["Model loss"])

    # plot training loss
    fig.add_trace(go.Scatter(x=list(range(1, len(history.history["loss"]) + 1)), y=history.history["loss"], mode='lines', name='Train'))

    if 'val_loss' in history.history:
        # plot validation loss if available
        fig.add_trace(go.Scatter(x=list(range(1, len(history.history["val_loss"]) + 1)), y=history.history["val_loss"], 
                                 mode='lines', name='Validation'))

    # set layout options
    fig.update_layout(
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Loss"),
        legend=dict(x=1.02, y=1, traceorder='normal', orientation='v'),
        width=800,  # adjust the width as needed
        height=400  # adjust the height as needed
    )

    # show figure
    fig.show()
    
def evaluate_model(model, x_test, y_test, scaler_target):
    '''
    This function evaluates the performance of the model using the test data. It plots the performance of the model on the test data and plots
    the actual closing values. It also calculates the mean absolute error, mean absolute percentage error, and median absolute percentage
    error for the actual and predicted closing values.

    @param model: Sequential model predicting stock price
    @param x_test: testing input sequence data
    @param y_test: testing target data
    @param scaler_target: MinMaxScaler for the target data

    The function does not return anything and instead plots the predicted and actual closing stock prices and prints out the three metrics. 
    '''
    y_predicted_close = model.predict(x_test) # predict closing prices
    y_predicted_close = scaler_target.inverse_transform(y_predicted_close) # unscale predictions
    y_true_close= scaler_target.inverse_transform(y_test.reshape(-1, 1))

    # Create traces for actual and predicted closing prices
    trace_actual = go.Scatter(x=np.arange(len(y_true_close)), y=y_true_close.flatten(),
                          mode='lines', name='Actual Closing Prices', line=dict(color='blue'))

    trace_predicted = go.Scatter(x=np.arange(len(y_predicted_close)), y=y_predicted_close.flatten(),
                             mode='lines', name='Predicted Closing Prices', line=dict(color='red'))

    # Create layout
    layout = go.Layout(title='Actual vs Predicted Closing Prices',
                   xaxis=dict(title='Days'),
                   yaxis=dict(title='Closing Prices'),
                   showlegend=True,
                   legend=dict(x=1, y=1, traceorder='normal', orientation='v'))

    # Create figure
    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)

    # Show the figure
    fig.show()

    # Mean Absolute Error (MAE)
    MAE = mean_absolute_error(y_true_close, y_predicted_close)
    print(f'Mean Absolute Error (MAE): {np.round(MAE, 2)}')

    # Mean Absolute Percentage Error (MAPE)
    MAPE = np.mean((np.abs(np.subtract(y_true_close, y_predicted_close) / y_true_close))) * 100
    print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE = np.median((np.abs(np.subtract(y_true_close, y_predicted_close) / y_true_close))) * 100
    print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')










