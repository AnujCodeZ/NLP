import nltk
import numpy as np
from nltk.corpus import twitter_samples

from preprocessing import preprocess, build_freqs

# Download all data utils
nltk.download('twitter_samples')
nltk.download('stopwords')

# Loading data
all_pos_tweets = twitter_samples.strings('positive_tweets.json')
all_neg_tweets = twitter_samples.strings('negative_tweets.json')

train_pos = all_pos_tweets[:4000]
test_pos = all_pos_tweets[4000:]
train_neg = all_neg_tweets[:4000]
test_neg = all_neg_tweets[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# Building frequency dict
freqs = build_freqs(train_x, train_y)

# Extracting features
def extract_features(tweet):
    words = preprocess(tweet)
    X = np.zeros((1, 3))
    X[0, 0] = 1 # bias
    
    for word in words:
        # Positive words
        X[0, 1] += (freqs[(word, 1.0)] if freqs.get((word, 1.0)) else 0)
        # Negative words
        X[0, 2] += (freqs[(word, 0.0)] if freqs.get((word, 0.0)) else 0)
    
    assert(X.shape == (1, 3))
    return X

# Data function
def load_data(train=False):
    x_set = None
    y_set = None
    if train:
        x_set = train_x
        y_set = train_y
    else:
        x_set = test_x
        y_set = test_y
    X = np.zeros((len(x_set), 3))
    for i in range(len(x_set)):
        X[i, :] = extract_features(x_set[i])
    
    return X, y_set
    