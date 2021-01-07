import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data.csv') 
X = df.iloc[:, 0:2].values
y = df.iloc[:, -1].values


firstDF = df[y == 0]
secondDF = df[y == 1]


 
def hypothesis(x, w):
    return np.sign(np.dot(w, x))

 
# Make predictions on all data points
# and return the ones that are misclassified.
def predict(hypothesis_function, X, y, w):  
    predictions = np.apply_along_axis(hypothesis_function, 1, X, w)
    misclassified = X[y != predictions]
    return misclassified

def pick_one_from(misclassified_examples, X, y):
    np.random.shuffle(misclassified_examples)
    x = misclassified_examples[0]
    index = np.where(np.all(X == x, axis=1))
    return x, y[index]



def perceptron_learning_algorithm(X, y):
    w = np.random.rand(3) # can also be initialized at zero.
    misclassified_examples = predict(hypothesis, X, y, w)
    while misclassified_examples.any():
        x, expected_y = pick_one_from(misclassified_examples, X, y)
        w = w + x * expected_y # update rule
        misclassified_examples = predict(hypothesis, X, y, w)
    return w

X_augmented = np.c_[np.ones(X.shape[0]), X]
w = perceptron_learning_algorithm(X_augmented, y)

print(w)