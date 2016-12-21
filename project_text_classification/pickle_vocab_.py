#!/usr/bin/env python3
import pickle


def main():
    vocab = dict()
    #idx = 0
    with open('./newData/vocab_cut_build_cut_.txt') as f:
        for idx,line in enumerate(f):
            vocab[line.strip()] = idx
            #print(len(vocab),idx)

    with open('./newData/vocab_.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
