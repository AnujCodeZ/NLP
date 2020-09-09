import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def preprocess(tweet):
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