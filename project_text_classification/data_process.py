import pandas as pd
import nltk
import pickle
from os import path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def main():
    #d = path.dirname('E:/2016-2017fall/pcml/projects/ML_Project2_nsy/')
    #doc_name = input('your path')
    #text = open('./newdata/vocab_cut.txt','r',encoding='utf-8')
    #text1 = open('./newdata/vocab_cut_.txt','w',encoding='utf-8')
    text = open('./twitter-datasets/train_pos_full.txt','r',encoding='utf-8')
    text1 = open('./newdata/train_pos_full_.txt','w',encoding='utf-8')
    
    s = text.readline()
    while(s):
        #if(s!="<user>\n" and s!="<url>\n"):
        # build token
        tokens= nltk.word_tokenize(s)
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
        #word = " ".join(lemmatized_content)
        word = " ".join(filtered_content)
        if(len(word)!=0):
            text1.write(word+'\n')
            #print(len(word))
        s = text.readline()

if __name__ == '__main__':
    main()