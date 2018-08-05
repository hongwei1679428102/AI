import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec


raw_text = open('corpus_text.txt').read()
raw_text = raw_text.lower()

# print(raw_text)

sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
sents = sentensor.tokenize(raw_text)
print(sents[:3])
corpus = []
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))

print(len(corpus))
print(corpus[:3])

# w2v乱炖
w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)

print(w2v_model['project'])