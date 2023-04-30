import nltk                         # NLP toolbox
from os import getcwd
import pandas as pd                 # Library for Dataframes
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt     # Library for visualization
import numpy as np                  # Library for math functions

from WordFrequencies import process_tweet, build_freq # Our functions for NLP

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = all_positive_tweets + all_negative_tweets ## Concatenate the lists.
labels = np.append(np.ones((len(all_positive_tweets),1)), np.zeros((len(all_negative_tweets),1)), axis = 0)

# split the data into two pieces, one for training and one for testing (validation set)
train_pos  = all_positive_tweets[:4000]
train_neg  = all_negative_tweets[:4000]

train_x = train_pos + train_neg

print("Number of tweets: ", len(train_x))

data = pd.read_csv("/Users/ozgeguney/PycharmProjects/pythonProject/data/logistic_features.csv"); # Load a 3 columns csv file using pandas function
print(data.head(10)) # Print the first 10 data entries

X = data[['bias', 'positive', 'negative']].values # Get only the numerical values of the dataframe
Y = data['sentiment'].values; # Put in Y the corresponding labels or sentiments

print(X.shape) # Print the shape of the X part
print(X) # Print some rows of X

theta = [6.03518871e-08, 5.38184972e-04, -5.58300168e-04]

