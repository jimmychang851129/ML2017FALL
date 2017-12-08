from gensim import corpora
from gensim import models
from gensim.models import word2vec
import numpy as np
import csv
from keras.preprocessing import text
from keras import callbacks
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.models import Model,Sequential
from keras.layers import Input, GRU, LSTM, Dense, Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
import sys,os
import itertools

alldata = []
trainx = []
trainy = []
testx = []
word_limit = 3  # how many word will be left(the top 20000 most freq word) 2w ->2w.5

pad_maxlength = 40
wordvec_size = 200

return_sequence = False
hidden_size = 256
dropout_rate = 0.65	# dropout 0.4
epochs = 30
batch=1000
validation = 0.2
dropout_layer = 0.7
##########
# config #
##########
EMBEDDING_DIM = 100
path = ".."
testfile = sys.argv[1]
outputcsv = sys.argv[2]

#############
# test used #
############
# path = ".."
# testfile = path+"/data/testtestdata.txt"
# trainfile = path+'/data/testdata.txt'
# outputcsv = path+"/output/word2vec.csv"
# filepath = path+"/output/word2vec{epoch:03d}-{val_acc:.3f}.hdf5"
# modelfile = path+'/output/model_label'

print("start reading testting file ...")
fo = open(testfile, "r",encoding='utf-8')
fo.readline()		# rid of index
while True:
	line = fo.readline().rstrip('\n')
	if not line:
		break
	alldata.append(line.split(',',1)[1])

print("finish reading testing file ...")


for i in range(len(alldata)):
	alldata[i] = text.text_to_word_sequence(alldata[i],filters="",split=" ")


print("start remove adjacent char...")
for i in range(len(alldata)):
	for j in range(len(alldata[i])):
		alldata[i][j] = ''.join(ch for ch, _ in itertools.groupby(alldata[i][j]))

print("finish remove adjacent char...")
wordmodel = word2vec.Word2Vec.load("vocab")
tmp = np.ndarray(shape=(1,wordvec_size))
tmp.fill(0)
weights = wordmodel.wv.syn0
weights = np.append(tmp,weights,axis=0)
vocab = dict([(k, (v.index)+1) for k, v in wordmodel.wv.vocab.items()])

testx = alldata
data = np.ndarray(shape=(len(testx),pad_maxlength))
data.fill(0)
for i in range(len(testx)):
	for j in range(len(testx[i])):
		if testx[i][j] in vocab:
			data[i][j] = vocab[testx[i][j]]
		else:
			data[i][j] = 0

model = Sequential()
model.add(Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights]))
model.add(LSTM(128, return_sequences=False,dropout=dropout_rate))
model.add(Dropout(dropout_layer))
# model.add(Dense(256,activation='relu'))
model.add(Dense(2,activation='softmax'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.load_weights("bestmodel.hdf5",by_name=False)

ans = model.predict(data)
ans = ans.argmax(axis=-1)
fw = open(outputcsv,'w')
fw.write('id,label\n')
for i in range(len(ans)):
    fw.write('{},{}\n'.format(i,ans[i]))
print("finish writing output")
