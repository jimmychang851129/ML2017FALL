# correct normalized(x_train+x_test normalized)

import csv
import numpy as np
import pandas as pd
from numpy.linalg import inv
import math
import sys
##################
#	   data      #
##################
testset = []
ans = []
label = []	#store Y_train label
alldata =[]
lr_rate = 0.05
iteration = 10000
bias = 0.001
weight = []
cnt = 0
index_nor = [0,1,3,4,5]

#############
#	init    #
#############
for i in range(106):
	weight.append(0.0)
weight = np.array(weight)

pregrad = []
for i in range(106):
	pregrad.append(0)
biasgrad = 0

##################
#	read file    #
##################
trainy = pd.read_csv(sys.argv[4],sep=',',header=0)
for index,row in trainy.iterrows():
	label.append(row[0])

######################
#	 noramalize      #
######################
def Normalize(x_train,x_test):
	normal_mean = np.zeros(106)
	normal_std = np.ones(106)
	npall = np.concatenate((x_train, x_test))
	print("npall = ",npall.shape)
	mean = (sum(npall) / npall.shape[0])
	variance = np.std(npall,axis=0)
	mean = np.tile(mean, (npall.shape[0], 1))
	variance = np.tile(variance, (npall.shape[0], 1))

	all_normal = (npall-mean)/variance
	normal = all_normal[0:x_train.shape[0]]
	npnormal = all_normal[x_train.shape[0]:]
	# print(normal)
	return normal,npnormal

#######################
#	 read all file    #
#######################
data = pd.read_csv(sys.argv[3],sep=',',header=0)
x_train = np.array(data.values)

test = pd.read_csv(sys.argv[5])
x_test = np.array(test.values)

##################
#	normalize    #
##################
normal,npnormal = Normalize(x_train,x_test)

# npnormal = Normalize(testset)
# print("npnormal = ",npnormal)

################
#	sigmoid    #
################
def sigmoid(z):
	result = 1/(1.0+np.exp(-z))
	return np.clip(result,1e-10,1-(1e-10))

######################
#	rmse function    #
######################
def rmse(normal,weight,bias,op,iteration):
	normal_rmse = np.dot(normal,weight)+bias
	normal_rmse = np.around(sigmoid(normal_rmse))
	result_rmse = (label == normal_rmse.squeeze())
	# print("result_ = ",result_rmse)
	print("progression %d/%d" % (op,iteration))
	print('Valid acc = %f' % (float(result_rmse.sum()) / result_rmse.shape[0]))


#################
#	training    #
#################
for i in range(iteration):
	L = np.dot(normal, weight.transpose()) +bias  ## big bug!!! not np.dot(normal,weight)
	# print(L)								 ## is np.dot(normal,weight.transpose())
	result = sigmoid(L)
	# print(result)
	loss = (result - label)
	biasgrad += (loss.sum()/(len(normal)))**2
	gradiant = np.dot(normal.transpose(),loss)
	pregrad += gradiant**2
	weight -= lr_rate*gradiant/np.sqrt(pregrad)
	bias -= lr_rate*loss.mean()/math.sqrt(biasgrad)
	if i % 1000 == 0:
		rmse(normal,weight,bias,i,iteration)
print("weight shape : ",weight.shape)
print("bias = ",bias)
# print("weight = ",weight)

###############
#	testing   #
###############
rmse(normal,weight,bias,iteration,iteration)

#########################
# 	calculate predict   #
#########################
cnt_true = 0
cnt_false = 0
print("testset shape : ",npnormal.shape)
testoutput = np.dot(npnormal,weight.transpose())+bias
# print("output = ",testoutput)
testresult = sigmoid(testoutput)

for i in testresult:
	if i >= 0.5:
		ans.append(1)
		cnt_true +=1
	else:
		ans.append(0)
		cnt_false +=1
print("cnt_true = ",cnt_true)
print("cnt_false = ",cnt_false)

#########################
# 	write outputfile    #
#########################
fw = open(sys.argv[6],"w")
csvCursor = csv.writer(fw)
writefile = ["id","label"]
csvCursor.writerow(writefile)
for i in range(1,16282):
	writefile[0] = str(i)
	writefile[1] =ans[i-1]
	csvCursor.writerow(writefile)

# #####################
# # 	write weight    #
# #####################
# fw = open("weight.csv","w")
# csvCursor = csv.writer(fw)
# csvCursor.writerow(np.append(weight,bias))
