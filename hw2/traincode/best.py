# best feat don't change any thin OAO
# correct normalized 110 feature with ln
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
lr_rate = 0.1
iteration = 5000
bias = 0.001
weight = []
cnt = 0
index_nor = [0,1,3,4,5]
index_ch = [0,3,4,5]
feature_cnt = 106+len(index_ch)*1

#############
#	init    #
#############
for i in range(feature_cnt):
	weight.append(0.0)
weight = np.array(weight)

pregrad = []
for i in range(feature_cnt):
	pregrad.append(0)
biasgrad = 0

for i in range(106,feature_cnt):
	index_nor.append(i)
print("index_nor = ",index_nor)

######################
#	 noramalize      #
######################
def Normalize(alldata,testset):
	cnt_alldata = len(alldata)
	tmp = cnt_alldata
	for i in range(len(testset)):
		alldata.append([])
		for j in testset[i]:
			alldata[cnt_alldata].append(j)
		cnt_alldata +=1
	normal_mean = np.zeros(feature_cnt)
	normal_std = np.ones(feature_cnt)
	npall = np.array(alldata)
	
	allmean = npall.mean(axis=0)
	# print("norma_mean = ",normal_mean[index_nor])
	normal_mean[index_nor] = allmean[index_nor]

	allstd = npall.std(axis=0)
	normal_std[index_nor] = allstd[index_nor]
	# print("allmean = ",allmean,"allstd = ",allstd)
	normal_all = (npall-normal_mean)/normal_std
	normal = normal_all[:tmp]
	npnormal = normal_all[tmp:]
	# print(normal)
	return normal,npnormal

#######################
#	 read all file    #
#######################
trainy = pd.read_csv(sys.argv[4],sep=',',header=0)	# y_train
for index,row in trainy.iterrows():
	label.append(row[0])

data = pd.read_csv(sys.argv[3])						# x_train

for index,row in data.iterrows():
	alldata.append([])
	for i in row:
		alldata[index].append(i)
	for i in index_ch:
		if alldata[index][i] != 0:
			alldata[index].append(math.log(alldata[index][i]))
		else:
			alldata[index].append(0)

data = pd.read_csv(sys.argv[5],delimiter = ',')		# x_test
for index,row in data.iterrows():
	testset.append([])
	for i in row:
		testset[index].append(i)
	for i in index_ch:
		if testset[index][i] != 0:
			testset[index].append(math.log(testset[index][i]))
		else:
			testset[index].append(0)
normal,npnormal = Normalize(alldata,testset)
print("normal shape : ",normal.shape)
print("npnormal shape : ",npnormal.shape)

##################
#	normalize    #
##################
print("normal shape : ",normal.shape)
print("npnormal shape : ",npnormal.shape)

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

