import tqdm
import numpy as np

from data import load_data
from preprocessing import preprocess, count_tweets, lookup


# Train
def train(freqs, train_x, train_y):
    log_likelihood = {}
    log_prior = 0
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]

    D = len(train_x)
    D_pos = sum(np.where(train_y > 0, 1., 0.))
    D_neg = D - D_pos

    log_prior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        log_likelihood[word] = np.log(p_w_pos / p_w_neg)

    return log_prior, log_likelihood

# Predict


def predict(tweet, log_prior, log_likelihood):
    words = preprocess(tweet)
    p = 0
    p += log_prior
    for word in words:
        if word in log_likelihood:
            p += log_likelihood[word]
    return p

# Test


def test(test_x, test_y, log_prior, log_likelihood):
    y_hats = []
    for tweet in test_x:
        if predict(tweet, log_prior, log_likelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)

    error = (y_hats == test_y)
    accuracy = np.mean(error)
    return accuracy

# Training
train_x, train_y = load_data()
freqs = count_tweets({}, train_x, train_y)
log_prior, log_likelihood = train(freqs, train_x, train_y)

# Testing
test_x, test_y = load_data(train=False)
accuracy = test(test_x, test_y, log_prior, log_likelihood)
print(f'Validation Accuracy: {accuracy}')

# Check yourself
tweet = 'I am happy'
print(predict(tweet, log_prior, log_likelihood))
