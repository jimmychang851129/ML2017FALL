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
bias = -0.546843085608
cnt = 0
index_nor = [0,1,3,4,5]
index_ch = [0,3,4,5]
feature_cnt = 106+len(index_ch)*1

#############
#	init    #
#############
weight = [-2.36455646354,0.0804499819318,0.889636955531,4.33239320513,0.736743271772,0.108470752267,-0.0377231137979,-0.681471121967,-3.50802175122,-0.458952105267,-0.279881998401,-0.90861854548,-0.810525911532,-4.40078560407,-0.511605742572,-1.24012681667,-1.15649729964,-0.819102225918,-1.82025738952,-1.49874079628,-1.60810281544,-1.40453508298,-0.0321447901818,-0.000859355798524,0.595888127618,1.67601040494,-0.496643561887,0.903109209592,-7.77378001706,1.47902754576,-0.152402889263,-0.777055490498,2.18351257855,1.21855221423,-0.727068765632,-0.989897424513,-0.919594799752,-0.287739129559,-0.240517307013,-1.37558364217,-0.222799585297,0.532250807344,-1.16897254266,-0.887229677835,-0.540076312297,-1.05353220178,-3.45511742992,0.266866087648,0.39347172237,0.0742682299903,0.403830527128,-0.366265928744,-0.526871598565,-0.388787519103,-0.100140362643,-0.835814572759,-1.07160879988,-0.290226782531,1.01898144523,-1.10782377272,-0.407355554611,-0.737731413746,-0.878310915202,-0.488836682189,0.631520420034,-0.290040550699,-1.40507759067,-2.51912938421,-0.252554274972,-2.53048764933,-0.844428525177,-1.15261052896,-0.256786441469,-0.0163230559496,-0.163906596966,-1.53979657738,-0.783002802095,-0.702510347672,-3.08669664441,-1.97316810267,-0.684074415531,-0.361768058527,-0.962793509453,-0.680670167297,0.196020373833,0.133271208168,-0.574505701288,-0.295809802577,-1.07113816382,-1.2201557304,-1.45470184322,-4.22114031113,-1.48871745669,-0.251221499957,-0.502200544199,-0.609527467999,-1.06906196037,-0.69080176316,-1.66444180024,-0.608354865772,-1.11593199973,-1.12870163086,-0.421097516342,-1.65917909204,0.0376240304513,-0.841365100724,3.09682121557,-0.526304291111,-0.49682019245,0.261282074363]
weight = np.array(weight)
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

# npnormal = Normalize(testset)
# print("npnormal = ",npnormal)

################
#	sigmoid    #
################
def sigmoid(z):
	result = 1/(1.0+np.exp(-z))
	return np.clip(result,1e-10,1-(1e-10))

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
