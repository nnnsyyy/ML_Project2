#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random


def main():
    print("loading cooccurrence matrix")
    with open('./newdata_/cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = cooc.max()
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 50
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 100
    gamma = 20
    
    min_mse=1000
    mse = 0
    best_xs = np.zeros((cooc.shape[0], embedding_dim))

    for epoch in range(epochs):
        pre_mse = mse
        mse = 0
        grad_xs = np.zeros((cooc.shape[0], embedding_dim))
        grad_ys = np.zeros((cooc.shape[1], embedding_dim))
				
        print("epoch {}".format(epoch)+" gamma {}".format(gamma))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            #scale = 2 * eta * fn * (logn - np.dot(x, y))
            mse_n = 1/2 * (logn - np.dot(x, y))**2 * (fn) * eta
            mse += mse_n
            grad_xs[ix,:] -= (logn - np.dot(x, y)) * fn * y * eta
            grad_ys[jy,:] -= (logn - np.dot(x, y)) * fn * x * eta
        print("mse {}".format(mse))
        xs -= gamma * grad_xs
        ys -= gamma * grad_ys
		
        if(min_mse > mse):
            min_mse = mse
            best_xs = xs
            
        if(epoch>=10):
            gamma = gamma*(1-1/(1+epoch*5))
            if(mse-min_mse > 1 or abs(pre_mse-mse) < 0.1):
                break
       
    print("best gamma {}".format(gamma))
    print("min_mse {}".format(min_mse))
    np.save('./newdata_/embeddings_dim50g10', best_xs)


if __name__ == '__main__':
    main()
