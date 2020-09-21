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

def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n

def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    for tweet, y in zip(tweets, ys):
        for word in preprocess(tweet):
            pair = (word, y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result