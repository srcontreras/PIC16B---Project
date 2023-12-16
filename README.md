# PIC16B---Project
In this project, we studied the potential effect of sentiment on the stock price of Starbucks. We use news articles for sentiment analysis test, and use the generated scores as input to the model on predicting stock price. We are satisfied with the result we achieved, and we hope you can enjoy our project as well.

It is highly recommended that you run everything in Google Colab. Our tutorial (final report) also has a thorough explanation of how to run our code in Colab. Therefore, please run all the codes there, instead of running in your local device. 

This is an overview of our project.
1. News Sentiment: We obtained our news articles through web scraping and APIs. We processed this information (tokenization, data augmentation, etc.) so that it could be fed into our news sentiment algorithm. The goal of our algorithm was to input a news article and receive a score from 0 to 2, representing negative, neutral, or positive sentiments.
2. Stock Price Model: Since we aim to predict stock prices, we needed historical stock prices of Starbucks. We utilized the Yahoo! Finance Python module to download stock price data starting from 2021. We cleaned this data, added features, split and normalized the data, and tested our model without sentiment. We employed recurrent neural networks (LSTM) to create our model.
3. Testing with Sentiment: With our stock price model operational and news article data sentimentalized, we incorporated sentiment into our dataset and tested our stock price model. We evaluated its performance using mean absolute error (MAE), mean absolute percentage error (MAPE), and median absolute percentage error (MDAPE). These metrics allowed us to see beyond just validation loss and accuracy.

For any specific details, please refer to the notebook we have above. 
