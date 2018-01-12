import numpy as np
from keras.models import Model
from keras.layers import Dense,Input
from sklearn import cluster
from keras import callbacks
from keras.callbacks import Callback
from keras.models import load_model
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import DistanceMetric
import sys
trainfile = sys.argv[1]
testfile = sys.argv[2]
outputfile = sys.argv[3]

epoch = 30
batch_size = 2000
encoding_dim = 40
testdata = []

data = np.load(trainfile)
mean = np.mean(data,axis=0)
std = np.std(data,axis=0)
data =data - mean

for i in range(784):
	if std[i] != 0:
		data[:,i] = data[:,i] / std[i]

# data /= 255

input_word = Input(shape=(784,))
encoded = Dense(512,activation="relu")(input_word)
encoded = Dense(256,activation="relu")(encoded)
encoded = Dense(128,activation="relu")(encoded)
encoder_output = Dense(encoding_dim)(encoded)

decoded = Dense(128,activation="relu")(encoder_output)
decoded = Dense(256,activation="relu")(decoded)
decoded = Dense(512,activation="relu")(decoded)
decoded = Dense(784,activation="relu")(decoded) # activation

autoencoder = Model(input=input_word,output=decoded)
encoder = Model(input=input_word,output=encoder_output)

autoencoder.compile(optimizer="adam",loss="mse")

earlystopping = callbacks.EarlyStopping(monitor='loss', patience = 3, verbose=1, mode='min')
callbacklist = [earlystopping]
autoencoder.fit(data,data,epochs=epoch,batch_size=batch_size,shuffle=True,callbacks=callbacklist)
autoencoder.save("./best_upload")
output = encoder.predict(data)

##########
# Kmeans #
##########
ans = cluster.KMeans(n_clusters = 2).fit(output)
ans_label = ans.labels_

testdata = []
with open(testfile,'r') as f:
	f.readline()
	for line in f:
		line = line.strip("\n").split(',')
		testdata.append([int(line[1]),int(line[2])])

final = []
for i in range(len(testdata)):
	if ans_label[testdata[i][0]] == ans_label[testdata[i][1]]:
		final.append(1)
	else:
		final.append(0)

fw = open(outputfile,'w')
fw.write('ID,ans\n')
for i in range(len(final)):
    fw.write('{},{}\n'.format(i,final[i]))