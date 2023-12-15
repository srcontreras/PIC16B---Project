import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers import TextVectorization

def TrainModel(df):
    le = LabelEncoder()
    df['Category'] = le.fit_transform(df['Sentiment'])
    
    dataset = make_dataset(df)
    dataset = dataset.shuffle(buffer_size = len(dataset), reshuffle_each_iteration=False)
    
    train_size = int(0.7*len(dataset))
    val_size = int(0.2*len(dataset))
    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size + val_size)
    
    max_tokens = 3000
    sequence_length = 50
    vectorize_layer = TextVectorization(standardize=remove_punc,
                                        max_tokens=max_tokens, # only consider this many words
                                        output_mode='int',
                                        output_sequence_length=sequence_length)