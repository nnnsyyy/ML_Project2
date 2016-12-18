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
                word_num=1
                sentence_vector = np.zeros(20)
                for t in line.strip().split(): 
                    if(vocab.get(t)!=None):
                        sentence_vector +=embed_word_metrix[vocab.get(t)-1,:]
                        word_num+=1
                sentence_vector = sentence_vector/word_num
                sentence_metrix.append(sentence_vector)
    return sentence_metrix

def build_sample(train_all, size):
    train_sample = train_all[np.random.randint(train_all.shape[0],size = size),:]
    y_train_sample = train_sample[:,0]
    x_train_sample = np.delete(train_sample,0,axis=1)
    return x_train_sample,y_train_sample
