import csv
import numpy as np
import random
import pandas as pd
import subprocess
import nltk
import pickle
from scipy.sparse import *
from os import path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def sentence_metrix(path,vocab,embed_word_metrix):
    sentence_metrix = []
    for fn in [path]:
        with open(fn) as f:
            for line in f:
                word_num=0
                sentence_vector = np.zeros(embed_word_metrix.shape[1])
                for t in line.strip().split(): 
                    if(vocab.get(t)!=None):
                        sentence_vector +=embed_word_metrix[vocab.get(t),:]
                        word_num+=1
                if(word_num!=0):
                    sentence_vector = sentence_vector/word_num
                else:
                    sentence_vector = np.zeros(embed_word_metrix.shape[1])  
                sentence_metrix.append(sentence_vector)
    return sentence_metrix

def build_train(vocab,embed_word_metrix):
    s_metrix_pos = sentence_metrix("./data/train_pos_full_.txt", vocab,embed_word_metrix)
    s_metrix_neg = sentence_metrix("./data/train_neg_full_.txt",vocab,embed_word_metrix)
    s_metrix_pos = np.asarray(s_metrix_pos)
    s_metrix_neg = np.asarray(s_metrix_neg)
    s_metrix_pos = np.insert(s_metrix_pos,0,1,axis = 1)
    s_metrix_neg = np.insert(s_metrix_neg,0,-1,axis = 1)
    s_me = np.concatenate((s_metrix_pos,s_metrix_neg))
    return s_me

def build_test(vocab,embed_word_metrix):
    s_me_test = sentence_metrix("./data/test_data_.txt",vocab,embed_word_metrix)
    s_me_test = np.asarray(s_me_test)
    return s_me_test


def build_sample(train_all, size):
    train_sample = train_all[np.random.randint(train_all.shape[0],size = size),:]
    y_train_sample = train_sample[:,0]
    x_train_sample = np.delete(train_sample,0,axis=1)
    return x_train_sample,y_train_sample

def build_predictlabel(y_predict):
    y_label =np.zeros((y_predict.shape[0]))
    for i in range(0,10000,1):
        if (y_predict[i]>0):
            y_label[i]  = 1
        else:
            y_label[i] = -1
    return y_label

def normallize(metrix):
    metrix_nor = np.zeros((metrix.shape[0],metrix.shape[1]))
    for column in range(metrix.shape[1]):
        max = np.amax(metrix[:,column])
        metrix_nor[:,column] = metrix[:,column]/max
    return metrix_nor

def data_process(path):
    text = open(path+'.txt','r',encoding = 'utf-8')
    text1 = open((path+'_.txt'),'w',encoding = 'utf-8')  
    s = text.readline()
    while(s):
        #token = s.split()
        # build token
        tokens= nltk.word_tokenize(s)
        # reduce short words
        #tokens_not_short = [t for t in tokens if len(t) > 2]
        # reduce not-letter
        tokens_started_with_letter = [t for t in tokens if t[0].isalpha()]
        # remove stop words -- <user> <url>
        stop_words = set(stopwords.words('english'))
        local_stop = {"user", "url"}
        stop_words = stop_words | local_stop
        filtered_content = [w for w in tokens_started_with_letter if not w in stop_words]
        # stemming
        #ps = PorterStemmer()
        #stemmed_content = [ps.stem(word) for word in filtered_content]
        # lemmatization
        #lemmatizer = WordNetLemmatizer()
        #lemmatized_content = [lemmatizer.lemmatize(word) for word in stemmed_content]
        word = " ".join(filtered_content)
        if(path =='./data/vocab_cut'):
            if(len(word)!=0):
                text1.write(word+'\n')
        else:
            text1.write(word+'\n')
        s = text.readline()

def pickle_vocab(path):
    vocab = dict()
    idx = 0
    with open(path) as f:
        for idx,line in enumerate(f):
            vocab[line.strip()] = idx

    with open('./data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

def build_cooc():
    with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    data, row, col = [], [], []
    counter = 1
    for fn in ['./data/train_pos_full_.txt', './data/train_neg_full_.txt']:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open('./data/cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

def glove_sgd():
    print("loading cooccurrence matrix")
    with open('./data/cooc.pkl', 'rb') as f:
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
            gamma = gamma*(1-1/(1+epoch * 2))
            if(mse-min_mse > 1 or abs(pre_mse-mse) < 0.1):
                break
       
    print("best gamma {}".format(gamma))
    print("min_mse {}".format(min_mse))
    np.save('./data/embeddings', best_xs)

def matrix_factorization(co_metrix):
    mf = NMF(n_components=500,init='random',random_state=0,max_iter=100,alpha=0.75,eta=0.001)
    mf.fit(co_metrix)
    word_vector = mf.transform(co_metrix)
    return word_vector
