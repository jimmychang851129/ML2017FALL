import csv
import numpy as np
import pandas as pd
from numpy.linalg import inv
import math
import sys
for i in range(7):
	print("i = ",i,sys.argv[i])

##################
#	   data      #
##################
li_true = []	#store age education_number
cnt_true = 0	#store >50k
cnt_false = 0   #store <=50k
li_false = [] 	#store age edufation_number <=50
mean_true,variance_true = 0,0
mean_false,variance_false = 0,0
total_variance = 0
testset = []
ans = []
label = []	#store Y_train label
alldata =[]
index_nor = [0,1,3,4,5]

##################
#	read file    #
##################
trainy = pd.read_csv(sys.argv[4])
data = pd.read_csv(sys.argv[3])
for index,row in trainy.iterrows():
	label.append(row[0])

######################
#	 noramalize      #
######################
def Normalize(alldata):
	normal_mean = np.zeros(106)
	normal_std = np.ones(106)
	npall = np.array(alldata)
	allmean = npall.mean(axis=0)
	# print("norma_mean = ",normal_mean[index_nor])
	normal_mean[index_nor] = allmean[index_nor]

	allstd = npall.std(axis=0)
	normal_std[index_nor] = allstd[index_nor]
	# print("allmean = ",allmean,"allstd = ",allstd)
	normal = (npall-normal_mean)/normal_std
	# print(normal)
	return normal

########################
#	read train file    #
########################
for index,row in data.iterrows():
	alldata.append([])
	for i in row:
		alldata[index].append(i)
normal = Normalize(alldata)

#################
#	training    #
#################
print("len normal : ",len(normal))
print("len normal[0] = ",len(normal[0]))
for i in range(len(normal)):
	if label[i] ==1:
		li_true.append([])	
		for j in (normal[i]):
			li_true[cnt_true].append(j)
		cnt_true += 1
	else:
		li_false.append([])
		for j in normal[i]:
			li_false[cnt_false].append(j)
		cnt_false += 1

print("cnt_true = ",cnt_true)
print("cnt_false = ",cnt_false)
np_li_true = np.array(li_true)
np_li_false = np.array(li_false)

mean_true = np_li_true.mean(axis = 0)
mean_false = np_li_false.mean(axis = 0)
# print("mean_true = ",mean_true)
# print("men_false = ",mean_false)

##################
#	 variance    #
##################
for i in range(len(normal)):
    if label[i] == 1:
        variance_true += np.dot(np.transpose([normal[i]-mean_true]),[normal[i]-mean_true])
    else:
        variance_false += np.dot(np.transpose([normal[i]-mean_false]),[normal[i]-mean_false])
variance_true/=cnt_true
variance_false/=cnt_false
# tmp = np_li_true - mean_true.transpose()
# variance_true = np.dot(tmp.transpose(),tmp)/cnt_true
# tmp = np_li_false - mean_false.transpose()
# variance_false = np.dot(tmp.transpose(),tmp)/cnt_false
# print("variance B : ",variance_false)
print("variance shape : ",variance_true.shape)

########################
#	 total_variance    #
########################
p_true = cnt_true/(cnt_true+cnt_false)
p_false = cnt_false/(cnt_true+cnt_false)
print("p_true : ",p_true,"p_false : ",p_false)
total_variance = variance_true*p_true+variance_false*p_false
print("total variance : ",total_variance)
print("variance shape : ",total_variance.shape)
inverse_variance = np.linalg.inv(total_variance)
# print("inverse : ",inverse_variance)

################
#	sigmoid    #
################
def sigmoid(z):
	result = 1/(1+np.exp(-z))
	return np.clip(result,1e-10,1-(1e-10))

########################
#	 weight and bias   #
########################
weight = np.dot((mean_true-mean_false).transpose(),inverse_variance)
bias = -0.5*((mean_true.transpose()).dot(inverse_variance).dot(mean_true))+((mean_false.transpose()).dot(inverse_variance).dot(mean_false))/2+math.log(cnt_true/cnt_false)
# print("weight = ",weight)
# bias = 1.30449625534
# print("bias = ",bias)

#############
#	 RMSE   #
#############
normal_rmse = np.dot(normal,weight)+bias
normal_rmse = np.around(sigmoid(normal_rmse))
result_rmse = (label == normal_rmse.squeeze())
print("result_ = ",result_rmse)
print('Valid acc = %f' % (float(result_rmse.sum()) / result_rmse.shape[0]))

##################
#	read file    #
##################
data = pd.read_csv(sys.argv[5],delimiter = ',')
alldata = []

######################
#	 noramalize      #
######################
for index,row in data.iterrows():
	alldata.append([])
	for i in row:
		alldata[index].append(i)
npnormal = Normalize(alldata)

#########################
# 	calculate predict   #
#########################
cnt_true = 0
cnt_false = 0
print("testset shape : ",npnormal.shape)
testoutput = np.dot(weight.transpose(),npnormal.transpose())+bias
testresult = sigmoid(testoutput)
print("test result length = ",len(testresult))
for i in testresult:
	if i > 0.5:
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