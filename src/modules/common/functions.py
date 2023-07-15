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
    
def cross_entropy_error(y, t, is_onehot=True):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
    
  batch_size = y.shape[0]
  if is_onehot:
    return -np.sum(t * np.log(y + 1e-7)) / batch_size 
  else:
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
  
