import numpy as np
from keras.models import Model
from keras.layers import Dense,Input
from sklearn import cluster
from keras import callbacks
from keras.callbacks import Callback
from keras.models import load_model
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

encoder = load_model("./encoder")

output = encoder.predict(data)
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
