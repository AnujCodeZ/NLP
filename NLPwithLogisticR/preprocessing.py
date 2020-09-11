import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def preprocess(tweet):
    """preprocess the data, it includes
    1. Removing symbols and hyperlinks
    2. Tokenize
    3. Remove stop words and punctuation
    4. Perform stemming

    Args:
        tweet (str): string containing a tweet

    Returns:
        List[str]: processed tweet
    """
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags( only # sign )
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweet
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True,
                               strip_handles=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stop_words and # remove stop words
            word not in string.punctuation): # remove punctuation
            stem_word = stemmer.stem(word) # stemming
            tweets_clean.append(stem_word)
    
    return tweets_clean

def build_freqs(tweets, ys):
    """Returns a dictionary containing frequencies of 
    positive and negative sentiments of words

    Args:
        tweets (List[str]): tweets
        ys (List[float]): 0 or 1 sentiments for tweets
    
    Returns:
        freqs (Dict[(str, int), int]): frequencies of words
    """
    yslist = np.squeeze(ys).tolist()
    
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in preprocess(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs
    