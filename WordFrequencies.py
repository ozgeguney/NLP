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
    freqs2={}
    for y, tweet in zip(ys_list, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs2[pair] = freqs2.get(pair, 0) + 1

    return freqs


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = all_positive_tweets + all_negative_tweets
print("Number of tweets: ", len(tweets))
labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))

dictionary = {'key1': 1, 'key2': 2}
dictionary['key3'] = -5
dictionary['key1'] = 0

print(dictionary)
print(dictionary['key2'])
#print(dictionary['key8'])


if 'key1' in dictionary:
    print("item found: ", dictionary['key1'])
else:
    print('key1 is not defined')

# Same as what you get with get
print("item found: ", dictionary.get('key8', -1))

freqs = build_freq(tweets, labels)


print(f'type(freqs) = {type(freqs)}')

print(f'len(freqs) = {len(freqs)}')

#print(freqs)

keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']

data = []
for word in keys:
    pos = 0
    neg = 0
    if (word, 1) in freqs:
        pos = freqs[(word, 1)]
    if (word, 0) in freqs:
        neg = freqs[(word, 0)]
    data.append([word, pos, neg])

print(data)

fig, ax = plt.subplots(figsize = (8, 8))

# convert positive raw counts to logarithmic scale. we add 1 to avoid log(0)
x = np.log([x[1] + 1 for x in data])

# do the same for the negative counts
y = np.log([x[2] + 1 for x in data])

# Plot a dot for each pair of words
ax.scatter(x, y)

# assign axis labels
plt.xlabel("Log Positive count")
plt.ylabel("Log Negative count")

# Add the word as the label at the same position as you added the points just before
for i in range(0, len(data)):
    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

ax.plot([0, 9], [0, 9], color = 'red') # Plot the red line that divides the 2 areas.
plt.show()