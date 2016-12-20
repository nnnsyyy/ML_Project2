import pandas as pd
import nltk
import pickle
from os import path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def main():
    d = path.dirname('E:/2016-2017fall/pcml/projects/ML_Project2_nsy/')
    doc_name = "./twitter-datasets/test_data"
    text = open(path.join(d+doc_name+'.txt'))
    text1 = open(path.join(d+doc_name+'_.txt'),'w')
    
    s = text.readline()
    while(s):
        #token = s.split()
        # build token
        tokens= nltk.word_tokenize(s)
        # reduce short words
        tokens_not_short = [t for t in tokens if len(t) > 2]
        # reduce not-letter
        tokens_started_with_letter = [t for t in tokens_not_short if t[0].isalpha()]
        # remove stop words -- <user> <url>
        stop_words = set(stopwords.words('english'))
        local_stop = {"user", "url"}
        stop_words = stop_words | local_stop
        filtered_content = [w for w in tokens_started_with_letter if not w in stop_words]
        # stemming
        ps = PorterStemmer()
        stemmed_content = [ps.stem(word) for word in filtered_content]
        # lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_content = [lemmatizer.lemmatize(word) for word in stemmed_content]
        word = " ".join(lemmatized_content)
        text1.write(word+'\r\n')
        s = text.readline()

if __name__ == '__main__':
    main()