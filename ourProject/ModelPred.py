import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers import TextVectorization
from matplotlib import pyplot as plt
from yahoo_fin import stock_info as si


def TrainModel(df, starbucks, make_dataset, remove_punc):
    '''
    This function trains the model on the textual dataframe we have. The function will return the model
    so that it could be used for prediction on the Starbucks news dataframe for sentiment analysis. 
    It will also return the prediction dataset 
    
    @param df: dataframe, the train data we have which will be used to create Tensor's dataset;
    @param starbucks: dataframe, contains news articles of Starbucks; we create the tensor's dataset for
                      it in the same way as df;
    @param make_dataset: func, function for creating Tensor's dataset, defined in TextDataPrep.py;
    @param remove_punc: func, function for removing punctuations, defined in TextDataPrep.py.
    
    @rvalue: model after training, which will be used for prediction;
             the prediction dataset we will use for prediction
    '''
    
    le = LabelEncoder() # label the sentiment, 0: negative, 1: neutral, 2: positive
    df['Sentiment'] = le.fit_transform(df['Sentiment'])
    
    dataset = make_dataset(df) # make tensor dataset
    dataset = dataset.shuffle(buffer_size = len(dataset), reshuffle_each_iteration=False)
    
    train_size = int(0.7*len(dataset)) 
    val_size = int(0.2*len(dataset))
    train = dataset.take(train_size) # train set
    val = dataset.skip(train_size).take(val_size) # validation set
    test = dataset.skip(train_size + val_size) # test set
    
    # creating starbucks dataset
    title = pd.DataFrame(starbucks['title'])
    title.rename(columns = {'title': "Sentence"}, inplace = True)
    title["Sentiment"] = 0 # add one column for the purpose of text vectorization layer
    title = make_dataset(title)
    
    max_tokens = 3000
    sequence_length = 50
    
    # text vectorization layer used for NPL later
    vectorize_layer = TextVectorization(standardize=remove_punc, 
                                        max_tokens=max_tokens, # only consider this many words
                                        output_mode='int',
                                        output_sequence_length=sequence_length)
    vectorize_layer.adapt(train.map(lambda x, y: x))
    
    def vectorize_text(text, label):
        '''
        This function helps vectorize the test, train, validation and prediction set, 
        which will be used in the function code below.
    
        @param text: str, the text that needs to be vectorized;
        @label: int, no use, we include it because our dataset also includes "label", which is the sentiment.
    
        @rvalues: text after vectorizing and the sentiment label.
        '''
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), [label]
    
    # vectorize the train, validation and test set.
    train_vec = train.map(vectorize_text)
    val_vec   = val.map(vectorize_text)
    test_vec = test.map(vectorize_text) 
    real = title.map(vectorize_text) # for prediction
    
    # model we create
    model = tf.keras.Sequential([layers.Embedding(max_tokens, output_dim = 10, name="embedding"),
                                 layers.Dropout(0.2), 
                                 layers.GlobalAveragePooling1D(),
                                 layers.Dropout(0.2),
                                 layers.Dense(3)])
    
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(train_vec, epochs = 10, validation_data = val_vec, callbacks=[callback], verbose = True)
    
    # visualize test and validation accuracy
    plt.plot(history.history["accuracy"], label = "training")
    plt.plot(history.history["val_accuracy"], label = "validation")
    plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
    plt.legend()
    
    print("Accuracy on Test Set: ", model.evaluate(test_vec)) # see accuracy on test set
    
    return model, real


def pred_scores(model, real, starbucks, check = False):
    '''
    This function finally creates the prediction score on each day and returns it, so that it could be used
    to predict stock price change. We should be careful that the date of the news article and the date of 
    stock price are not aligned: sometimes on one day we have news articles but no stock price, sometimes vice versa.
    We need to handle this. 
    
    @param model: model that is used for prediction, returned by function TrainModel;
    @param real: the prediction dataset we have, returned by function TrainModel;
    @param starbucks: dataframe, contains news articles of starbucks;
    @check: bool, check accuracy of our prediction; If True, print out 5 predictions; Else, nothing happens
    
    @rvalue: list, the final aligned sentiment score on each day that could be used for price prediction.
    '''
    
    real_sentiment = model.predict(real).argmax(axis=1) # prediction
    starbucks['score1'] = real_sentiment # store the prediction
    
    if check: # check accuracy of prediction
        sentiment = starbucks[:5][["title", "score1"]]
        for i in range(5):
            print('"', sentiment.iloc[i]["title"], '"' , "scores: ", sentiment.iloc[i]["score1"])
            print('\n')
    
    pred = starbucks[["date", "score1"]]
    pred = pred.groupby("date").apply(np.mean) # calculate average sentiment score on each day
    pred = pred.reset_index()

    company = si.get_data('sbux') # get stock information
    company = company.loc["2021-01-04":] 
    company["date"] = company.index
    company = company.reset_index()
    company.drop(columns = "index", inplace = True)
    
    date = company[company['date'] < "2023-11-15"]
    date = date['date'].dt.strftime('%Y-%m-%d') # transform time to string
    date = pd.DataFrame(date)
    
    # because date for stock price and date for news articles are not aligned, we need to work on this
    date1 = date['date'].to_numpy()
    date2 = pred["date"].to_numpy()
    score = pred["score1"].to_numpy()
    
    score_pred = []
    memory = 0
    count = 0

    for i in range(len(score)):
        if i < len(date1) and date1[i] not in date2: # if on this day, we have stock price but no news article
            score_pred.append(1) # we deem it as neutral
            continue
            
        if date2[i] not in date1: # if on this day, we have news article but no stock price
            memory += score[i] # we calculate the average
            count += 1
            continue 
        
        # when we don't have stock price on the day but we do have scores,
        # we calculate the average
        if memory != 0:
            score_append = (memory + score[i]) / (count+1)
            score_pred.append(score_append)
            memory = 0
            indic = 0
            continue
    
        score_pred.append(score[i]) # if on this day, we have both stock price and news articles
    
    return score_pred
