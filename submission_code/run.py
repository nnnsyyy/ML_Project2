import numpy as np
import utilities
import sklearn
import subprocess
#Build vocabulary
print("start build vocab")
subprocess.call("./build_vocab.sh")
subprocess.call("./cut_vocab.sh")
utilities.data_process('./data/vocab_cut')
subprocess.call("./rebuild_vocab.sh")
subprocess.call("./recut_vocab.sh")

#Pre-process the text data
print("start pre_process data")
utilities.data_process('./data/train_pos_full')
utilities.data_process('./data/train_neg_full')
utilities.data_process('./data/test_data')

#Train the model
print("train model")
utilities.pickle_vocab('./data/vocab_rebuild_recut.txt')
vocab = np.load('./data/vocab.pkl')
utilities.build_cooc()
co_metrix = np.load('./data/cooc.pkl')
word_vector=utilities.matrix_factorization(co_metrix)


#Prediction
print("prediction")
train_all = utilities.build_train(vocab,word_vector)
sample_x,sample_y = utilities.build_sample(train_all,250000)
test_all = utilities.build_test(vocab,word_vector)

reg = sklearn.linear_model.LinearRegression()
reg = reg.fit(sample_x,sample_y)
y_label = reg.predict(test_all)
y_predict = utilities.build_predictlabel(y_label)

OUTPUT_PATH = './data/bestSubmission.csv' 
utilities.create_csv_submission(range(1,10001,1), y_predict, OUTPUT_PATH)