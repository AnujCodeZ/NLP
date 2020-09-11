import tqdm
import numpy as np

from data import load_data, extract_features

# Initialize parameters
theta = np.zeros((3, 1))

# Forward pass
def forward(x, theta):
    z = np.dot(x, theta)
    a = 1. / (1. + np.exp(-z))
    return a

# Cost function
def compute_cost(a, y):
    m = y.shape[0]
    loss = (-1/m) * (np.dot(y.T, np.log(a)) + 
                     np.dot((1 - y).T, np.log(1 - a)))
    return loss

# Backward pass
def backward(x, y, a):
    return np.dot(x.T, (a - y)) / x.shape[0]

# Train
def train(x, y, theta, alpha, num_iters):
    with tqdm.trange(num_iters) as t:
        for _ in t:
            a = forward(x, theta)
            loss = compute_cost(a, y)
            theta = theta - alpha * backward(x, y, a)
            t.set_description(f'Loss: {np.squeeze(loss):.3f}')
    return loss, theta

# Loading train data
x_train, y_train = load_data(train=True)

# Training
loss, theta = train(x_train, y_train, theta, 1e-9, 1500)

# Test
def test(x, y, theta):
    a = forward(x, theta)
    y_hat = np.where(a > 0.5, 1., 0.)    
    return np.mean(y_hat == y_test)

# Loading test data
x_test, y_test = load_data(train=False)

# Testing
accuracy = test(x_test, y_test, theta)
print(f'Accuracy: {accuracy}')

# Predict your own tweet
def predict(tweet, theta):
    x = extract_features(tweet)
    pred = forward(x, theta)
    return ('Positive' if pred > 0.5
            else 'Negative')

my_tweet = 'I am happy'
print(predict(my_tweet, theta))