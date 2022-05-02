
import numpy as np

seed = 0
np.random.seed(seed)

# Machine repairs
def uniform(a, b):
    return np.random.uniform(a, b)

def choice(list, weight, k):
    return np.random.choices(list, weights=weight, k=k)

# Machine breakdowns
def expovariate(value):
    return np.random.exponential(value)
    # return np.random.gamma()

def weibul(value):
    return np.random.weibull()

def gamma(value):
    return np.random.gamma()

