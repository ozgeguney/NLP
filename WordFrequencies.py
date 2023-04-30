import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import numpy as np
import re
import string


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_eng = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean= []
    for word in tweet_tokens:
        if word not in stopwords_eng and word not in string.punctuation:
            stemmed_word = stemmer.stem(word)
            tweets_clean.append(stemmed_word)
    return tweets_clean

new_tweet =process_tweet("@my name is özge, today I am happy and I feel good!!! :)")
print(new_tweet)

def build_freq(tweets, ys):
    #ys: an m x 1 array with the sentiment label of each tweet (either 0 or 1)
    # Convert np array to list since zip needs an iterable.
    ys_list = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(ys_list, tweets):
        for word in process_tweet(tweet):
            pair = (word,y)
            if pair in freqs:
                freqs[pair] +=1
            else:
                freqs[pair] = 1
    return freqs

ys = np.ones((5))
tweets = ['@my name is özge, today I am happy and I feel good!!! :) happi happy be happy']
a = build_freq(tweets, ys)
print(a)