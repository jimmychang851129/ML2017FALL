import pandas
import numpy as np
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Dropout,Activation,Flatten,Reshape
from keras.layers.merge import dot,add
from keras import callbacks
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import Callback
from sklearn.utils import shuffle
import sys,os

def rmse(y_true, y_pred):
	y_pred = K.clip(y_pred, 1., 5.)
	return K.sqrt(K.mean(K.square((y_true - y_pred))))

# testfile = "../data/test.csv"
# outputcsv = "./valid.csv"
testfile = sys.argv[1]
outputcsv = sys.argv[2]

n_users = 6040
n_movies = 3952
dim = 150
droprate = 0.5

u_input = Input(shape=[1])
userinput = Embedding(n_users, dim)(u_input)
userinput = Reshape((dim,))(userinput)
userinput = Dropout(droprate)(userinput)

m_input = Input(shape=[1])
movieinput = Embedding(n_movies, dim)(m_input)
movieinput = Reshape((dim,))(movieinput)
movieinput = Dropout(droprate)(movieinput)

user_bias = Embedding(n_users, 1)(u_input)
user_bias = Flatten()(user_bias)
movie_bias = Embedding(n_movies, 1)(m_input)
movie_bias = Flatten()(movie_bias)

out = dot([userinput, movieinput], -1)
out = add([out, user_bias, movie_bias])

model = Model(inputs=[u_input,m_input],outputs=out)
# model.summary()

model.compile(loss='mse',optimizer='adam',metrics=[rmse])

model.load_weights("./bestmodel.hdf5",by_name=False)

print("start reading testdata ... ")
testdata = pandas.read_csv(testfile, engine='python',header=None,skiprows=1,
                          sep=',', names=['testid','userid', 'movieid'])

print("finish reading testdata ... ")

testuser = testdata.userid-1
testmovie = testdata.movieid-1
ans = model.predict([testuser,testmovie])
# ans = ans*std+mean
ans = ans.clip(1.,5.)

print("start writing output ...")
fw = open(outputcsv,'w')
fw.write('TestDataID,Rating\n')
for i in range(len(ans)):
	fw.write('{},{}\n'.format(i+1,ans[i][0]))

print("finish writing output ...")
