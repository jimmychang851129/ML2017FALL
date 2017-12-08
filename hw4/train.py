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

csv.field_size_limit(sys.maxsize)

#############
#	var 	#
#############
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

# path = os.environ.get('GRAPE_DATASET_DIR')
# path = "../.."
# trainfile = path+"/data/training_label.txt"
# testfile = path+"/data/testing_data.txt"
# outputcsv = "output/word2vec.csv"
# filepath="output/word2vec{epoch:03d}-{val_acc:.3f}.hdf5"
# modelfile = 'output/model_label'
# unlabelfile = path+"/data/training_nolabel.txt"

trainfile = sys.argv[1]
unlabelfile = sys.argv[2]
filepath = "./word2vec{epoch:03d}-{val_acc:.3f}.hdf5"
#############
# test used #
############
# path = ".."
# testfile = path+"/data/testtestdata.txt"
# trainfile = path+'/data/testdata.txt'
# outputcsv = path+"/output/word2vec.csv"
# filepath = path+"/output/word2vec{epoch:03d}-{val_acc:.3f}.hdf5"
# modelfile = path+'/output/model_label'

####################
# read trainlabel  #
####################
print("start reading training file ...")
with open(trainfile,'r',encoding='utf-8') as f:
	for line in f:
		lines = line.strip().split(' +++$+++ ')
		alldata.append(lines[1])
		trainy.append(int(lines[0]))

print("finish reading training file ...")

###################
#	read unlabel  #
###################
print("start reading unlabel file ...")
with open(unlabelfile,'r',encoding='utf-8') as f:
	for line in f:
		alldata.append(line)

print("finish reading unlabel file ...")

print("start cutting word ...")
for i in range(len(alldata)):
	alldata[i] = text.text_to_word_sequence(alldata[i],filters="",split=" ")
print("finish cutting word ...")
print("alldata len = ",len(alldata))

########################
# remove adjacent char #
########################
print("start remove adjacent char...")
for i in range(len(alldata)):
	for j in range(len(alldata[i])):
		alldata[i][j] = ''.join(ch for ch, _ in itertools.groupby(alldata[i][j]))

print("finish remove adjacent char...")

print("start splitting data ...")

trainx = alldata[:len(trainy)]
trainy = np_utils.to_categorical(trainy, 2)

print("finish splitting data ...")
##################
#	build model  #
##################
print("start training model ...")

model = word2vec.Word2Vec(alldata,size=wordvec_size,min_count=word_limit,workers=4)

#########################
# get weights and index #
#########################
tmp = np.ndarray(shape=(1,wordvec_size))
tmp.fill(0)
weights = model.wv.syn0
weights = np.append(tmp,weights,axis=0)

vocab = dict([(k, (v.index)+1) for k, v in model.wv.vocab.items()])
# model.save(modelfile+"_0")
print("finish training model ...")
print("start turning word to index ...")
# words = list(model.wv.vocab)


data = np.ndarray(shape=(len(trainx),pad_maxlength))
data.fill(0)
for i in range(len(trainx)):
	for j in range(len(trainx[i])):
		if trainx[i][j] in vocab:
			data[i][j] = vocab[trainx[i][j]]
		else:
			data[i][j] = 0

print("finish turning word to index ...")
print("len trainx = ",len(data))
print("shape = ",data[0].shape)

############
# callback #
############
Checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max', period=1,save_weights_only=True)

batch_print_callback = callbacks.LambdaCallback(
    on_epoch_end=lambda batch, logs: print(
        'Epoch[%d] Train-accuracy=%f  Epoch[%d] Validation-accuracy=%f' %(batch, logs['acc'], batch, logs['val_acc'])))
earlystopping = callbacks.EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')
callbacklist = [Checkpoint,batch_print_callback,earlystopping]

#########
# model #
#########
model = Sequential()
model.add(Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights]))
model.add(LSTM(128, return_sequences=False,dropout=dropout_rate))
model.add(Dropout(dropout_layer))
# model.add(Dense(256,activation='relu'))
model.add(Dense(2,activation='softmax'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# fit the model
model.fit(data, trainy,batch_size=batch,epochs=epochs,validation_split=validation,shuffle=True,verbose=0,callbacks=callbacklist)