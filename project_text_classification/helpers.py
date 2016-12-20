import csv
import numpy as np
import random

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
                sentence_vector = np.zeros(20)
                for t in line.strip().split(): 
                    if(vocab.get(t)!=None):
                        sentence_vector +=embed_word_metrix[vocab.get(t)-1,:]
                        word_num+=1
                if(word_num!=0):
                    sentence_vector = sentence_vector/word_num
                else:
                    sentence_vector = np.zeros(20)  
                sentence_metrix.append(sentence_vector)
    return sentence_metrix

def build_train(vocab,embed_word_metrix):
    s_metrix_pos = sentence_metrix("./twitter-datasets/train_pos.txt", vocab,embed_word_metrix)
    s_metrix_neg = sentence_metrix("./twitter-datasets/train_neg.txt",vocab,embed_word_metrix)
    s_metrix_pos = np.asarray(s_metrix_pos)
    s_metrix_neg = np.asarray(s_metrix_neg)
    s_metrix_pos = np.insert(s_metrix_pos,0,1,axis = 1)
    s_metrix_neg = np.insert(s_metrix_neg,0,-1,axis = 1)
    s_me = np.concatenate((s_metrix_pos,s_metrix_neg))
    return s_me


def build_sample(train_all, size):
    train_sample = train_all[np.random.randint(train_all.shape[0],size = size),:]
    y_train_sample = train_sample[:,0]
    x_train_sample = np.delete(train_sample,0,axis=1)
    return x_train_sample,y_train_sample
