import numpy as np

def ordered_set(x):
    res,ind = np.unique(x, return_index=True)
    return res[np.argsort(ind)]

def get_alias(preal, palias):
     return abs(1/((1/preal) - 1*(1/palias)))
