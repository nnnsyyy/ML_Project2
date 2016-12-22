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
    s_metrix_pos = sentence_metrix("./newdata_/train_pos_full_.txt", vocab,embed_word_metrix)
    s_metrix_neg = sentence_metrix("./newdata_/train_neg_full_.txt",vocab,embed_word_metrix)
    s_metrix_pos = np.asarray(s_metrix_pos)
    s_metrix_neg = np.asarray(s_metrix_neg)
    s_metrix_pos = np.insert(s_metrix_pos,0,1,axis = 1)
    s_metrix_neg = np.insert(s_metrix_neg,0,-1,axis = 1)
    s_me = np.concatenate((s_metrix_pos,s_metrix_neg))
    return s_me

def build_test(vocab,embed_word_metrix):
    s_me_test = sentence_metrix("./newdata_/test_data_.txt",vocab,embed_word_metrix)
    s_me_test = np.asarray(s_me_test)
    return s_me_test


def build_sample(train_all, size):
    train_sample = train_all[np.random.randint(train_all.shape[0],size = size),:]
    y_train_sample = train_all[:,0]
    x_train_sample = np.delete(train_all,0,axis=1)
    return x_train_sample,y_train_sample

def normallize(metrix):
    metrix_nor = np.zeros((metrix.shape[0],metrix.shape[1]))
    for column in range(metrix.shape[1]):
        max = np.amax(metrix[:,column])
        metrix_nor[:,column] = metrix[:,column]/max
    return metrix_nor

