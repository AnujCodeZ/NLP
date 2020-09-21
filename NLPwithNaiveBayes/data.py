import nltk
import numpy as np
from nltk.corpus import twitter_samples


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

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

def load_data(train=True):
    if train:
        return train_x, train_y
    else:
        return test_x, test_y