import numpy as np

def softmax(a):
  C = np.max(a)
  exp_a = np.exp(a - C)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y

def identity(x):
  return x

def relu(x):
  return np.maximum(0, x)

def sigmoid(x):
    return np.piecewise(
        x,
        [x > 0],
        [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
    )