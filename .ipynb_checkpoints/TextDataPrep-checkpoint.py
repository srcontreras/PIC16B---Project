import numpy as np
import pandas as pd
import re
import string
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('stopwords')
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer # lemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# data augmentation
import nlpaug.augmenter.word as naw
import nlpaug.flow as nafc
from nlpaug.util import Action


def prepData(fin_data, positive, print_size = False):
    '''
    This function creates the dataframe that is for training the model. It has 2 parts: one is the 'fin_data' table
    and the other is the 'positive' table. We only take half of the 'positive' table to balance the "positive" 
    sentiment and the "neutral" sentiment. 
    
    @param fin_data: dataframe, contains financial sentences which are categorized into negative, 
                     neutral and positive sentiments.
    @param positive: dataframe, contains all positive news articles by web scraping. 
    @print_size: bool, if True, see how many texts we have for each sentiment;
                       if False, nothing happens; default is False
    
    @rvalue: dataframe, the raw dataframe for training purpose.
    '''
    
    positive['Sentiment'] = 'positive'
    rows = positive.shape[0]
    positive = positive.sample(frac=1) 
    merge = positive.iloc[:int(rows/2)] # only consider half of the positive table
    title = merge[['title', 'Sentiment']]
    title.rename(columns = {'title': 'Sentence'}, inplace = True)
    data = pd.concat([fin_data, title], axis = 0) # concatenate 2 tables
    data = data_t.reset_index()
    data.drop(columns = 'index', inplace = True)
    data = data.sample(frac=1)
    
    if print_size:
        print(data_t.groupby("Sentiment").apply(len))
    
    return data


def negative_aug(df):
    '''
    ***** Note *****
    This function would be very slow. Therefore, instead of running it, we have a negative_augmented_version.csv
    file that contains augmented texts. It's better to read that file directly. 
    ****************
    
    If we look at the number of texts for negative sentiment, which could be done by calling
    prepData(fin_data, positive, print_size = True), we may notice that it is much smaller than
    the other 2 sentiments. This might lead to bias in training. This function avoids this by adding
    data augmentation to negative sentences. 3 ways are used. After this, the numbers of texts for
    the 3 sentiments are similar.
    
    @param df: dataframe, the dataframe that is returned by function prepData(fin_data, positive, print_size)
    @rvalue: dataframe, augmented version.
    '''
    
    # 3 ways of augmentation
    aug1 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
    aug2 = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
    aug3 = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute")
    
    negative = data_t[data_t["Sentiment"] == "negative"] # all negative sentences
    
    negative_aug1 = negative["Sentence"].apply(aug1.augment) # apply augmentation
    negative_aug2 = negative["Sentence"].apply(aug2.augment) 
    negative_aug3 = negative["Sentence"].apply(aug3.augment) 
    
    # clean the result and concatenate to the original df to make a augmented version
    for i in [negative_aug1, negative_aug2, negative_aug3]:
        negative_i = [m[0] for m in i]
        negative_i = pd.DataFrame(negative_i)
        negative_i.rename(columns = {0: "Sentence"}, inplace = True)
        negative_i["Sentiment"] = "negative
        df = pd.concat([df, negative_i], axis = 0)
        
    return df # if you run this function, it's better to save this df, instead of re-running this function.


def remove_punc(input_data):
    '''
    This function removes the punctuation in the text. This will be used in the TextVectorization layer of the model.
    
    @param input_data: tf str, the text data of our dataset.
    
    @rvalue: tf str, the text after removing punctuations.
    '''
    
    no_punctuation = tf.strings.regex_replace(input_data,
                                             '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation


def get_wordnet_pos(treebank_tag):
    '''
    This function converts a Penn Treebank part-of-speech tag to a WordNet POS tag.
    This is part of the function defined next. 
    
    @param treebank_tag: str, a part of speech tag in Penn Treebank format.
    
    @rvalues: str or None, the corresponding WordNet part-of-speech tag (one of ADJ, VERB, NOUN, ADV),
                           or None if the treebank tag doesn't satisfy some conditions.
    '''
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemma(x):
    '''
    Lemmatize a sentence using NLTK's WordNetLemmatizer, converting each word to its base or dictionary form.
    This function tokenizes the input sentence, assigns part-of-speech tags, and then lemmatizes each word
    based on its POS tag. If a POS tag is not recognized, it defaults to NOUN.
    This function will be used with .apply() method.

    @param x: str, sentence in string format to be lemmatized.
    
    @rvalues: str, the lemmatized sentence, with each word in its base form.
    '''
    
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(x)
    tagged = nltk.pos_tag(tokens)
    lemmatized_sentence = []
    
    for word, tag in tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
        
    return ' '.join(lemmatized_sentence) # return lemmatized sentence of each entry of dataframe.


def make_dataset(df):
    '''
    This function transforms a pandas dataframe into a tensorflow dataset, with all the cleaning processes
    we defined above. 
    
    @param df: dataframe, contains the news article
    
    @rvalues: tensorflow dataset, contains clean text data.
    '''
    
    stop = stopwords.words("english")
    
    # change into lower case first
    df['cleaned'] = df['Sentence'].apply(lambda x: x.lower())
    
    # remove stopwords
    df['cleaned'] = df['cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
    
    # stemming/lemmatizing
    df['cleaned'] = df['cleaned'].apply(lambda x: lemma(x))
    
    data = tf.data.Dataset.from_tensor_slices((df['cleaned'], df['Category']))
    
    return data