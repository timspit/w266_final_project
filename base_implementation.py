# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#imports

import pandas as pd
import numpy as np
from common import utils
from nltk.util import ngrams

#load training data
twitter_train = pd.read_excel('./StanceDataset/train.xlsx')

#carve out the training tweets
tweets = twitter_train['Tweet']

#carve out training labels and convert to labels for feeding sklearn SVM
stance_labels = np.array(twitter_train['Stance'].apply(lambda x: 2 if x == "FAVOR" else 
                             (1 if x == "NONE" else 0)))

#tokenize tweets
tweets = tweets.apply(lambda x: x.split(" "))

#loop to generate uni, bi, and trigrams and store in list
word_ngrams = []

#inefficient nested for loops to create the universe of word ngrams (uni - tri)
for tweet in tweets:
    for n in range(1,4):
        for gram in ngrams(tweet,n):
            word_ngrams.append(gram)

#turn into unique list
word_ngrams = np.unique(np.asarray(word_ngrams)).tolist()

#create a dataframe to hold ngram feature 1 hot encodings
stance_features = pd.DataFrame()

## helper function to identify if n gram is in the string
def identify_ngram(string, ngram):
    counter = 0
    for gram in ngram:
        if gram in string:
            counter+=1
    if counter == len(ngram):
        return True
    else:
        return False

#fill in the feature matrix
for ngram in word_ngrams:
    stance_features[ngram] = tweets.apply(lambda x: 1 if identify_ngram(x,ngram) else 0)