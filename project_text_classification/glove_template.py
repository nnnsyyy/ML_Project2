#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random


def main():
    print("loading cooccurrence matrix")
    with open('./newdata/cooc.pkl', 'rb') as f:
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
	
	min_mse=150
    mse = 0
	best_xs = np.zeros((cooc.shape[0], embedding_dim))

    for epoch in range(1,epochs+1):
		pre_mse = mse
        grad_xs = np.zeros((cooc.shape[0], embedding_dim))
        grad_ys = np.zeros((cooc.shape[1], embedding_dim))
				
        print("epoch {}".format(epoch)+"gamma {}".format(gamma))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            mse_n = (scale/2)**2/(fn*eta)
            mse += mse_n
            grad_xs[ix,:] += -scale * y
            grad_ys[jy,:] += -scale * x
        print("mse {}".format(mse))
        xs += -gamma * grad_xs
        ys += -gamma * grad_ys
		
        if(epochs>10):
		    gamma = gamma*(1-1/(1+epoch*0.5))
            if(mse-min_mse > 10 or abs(pre_mse-mse) < 0.05):
                break
            else:
			    if(min_mse > mse):
			        min_mse = mse
				    best_xs = xs
                
	print("best gamma {}".format(gamma))
    print("min_mse {}".format(min_mse))
    np.save('./newdata/embeddings', best_xs)


if __name__ == '__main__':
    main()
