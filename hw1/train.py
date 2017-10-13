#################################
#   3項(pm2.5,pm10,O3)各二次式    #
#################################

import pandas as pd
import numpy as np
import math
import csv

data = pd.read_csv("./train.csv",encoding='big5')

#############
#   Data    #
#############
cnt_row = 0
Data = []
train_x = []
train_y = []
test_x = []
test_y = []
st_train = 0
ed_train = 12
st_test = 12
ed_test = 12
datapermonth = 480
parameter_cnt = 54	# 2-dimension
times = 4
iteration = 100000
lr_rate = 0.1
weight = []
data_per_month = 480
prev_grad = 0

#############
# read file #
#############
for i in range(18):
	Data.append([])

for index,row in data.iterrows():
	for i in range(3,27):
		if row[i] != "NR":
			Data[cnt_row%18].append(float(row[i]))
		else:
			Data[cnt_row%18].append(float(0))
	cnt_row +=1
print("Data length : ",len(Data))

######################
# read data to train #
######################
for i in range(st_train,ed_train):
	for j in range(data_per_month-9):			#data per month 20 days each month,20*24(hr) data -9（每次連續取9筆資料,跨天數)
		train_x.append([1])
		tmp = (data_per_month-9)*(i-st_train)+j
		tmp1 = data_per_month*i+j
		for t in range(7,10):
			for s in range(9):
				train_x[tmp].append(Data[t][tmp1+s])
				train_x[tmp].append(Data[t][tmp1+s]**2)
		train_y.append(Data[9][tmp1+9])
print("trainx : ",train_x[0])
print("trainy : ",train_y[0])

######################
# read data to test  #
######################
# for i in range(st_test,ed_test):
# 	for j in range(data_per_month-9):			#data per month 20 days each month,20*24(hr) data -9（每次連續取9筆資料,跨天數)
# 		test_x.append([1])
# 		tmp = (data_per_month-9)*(i-st_test)+j
# 		tmp1 = data_per_month*i+j
# 		for t in range(7,10):
# 			for s in range(9):
# 				test_x[tmp].append(Data[t][tmp1+s])
# 				test_x[tmp].append(Data[t][tmp1+s]**2)
# 		test_y.append(Data[9][tmp1+9])

# print("testset")
# print("x : ",test_x[0])
# print("y : ",test_y[0])

##########################
# 	regression function  #
##########################
def training(trainx,trainy,lr,w):
	global prev_grad
	L = trainx.dot(w)-trainy 				#Loss function
	gradiant = 2*np.dot(np_x.transpose(),L)
	print("gradiant : ",gradiant)
	prev_grad += gradiant**2
	w -= gradiant/np.sqrt(prev_grad)*lr
	return w,L

###########
#  init   #
###########
for i in range(parameter_cnt+1):
	weight.append(0.0)

#################
# 	regression  #
#################
np_x = np.array(train_x)
np_y = np.array(train_y)
print("train_x count : ",len(np_x))
print("train_y count : ",len(np_y))

for j in range(iteration):
	weight,loss = training(np_x,np_y,lr_rate,weight)
	if j%10000 ==0:
		print("loss : ",np.average(loss))
print("weight = ",weight,"loss = ",np.average(loss))

#################
# 	testset     #
#################
# print("### testset ###")
# nptest_x = np.array(test_x)
# nptest_y = np.array(test_y)
# final = np.dot(nptest_x,weight)-nptest_y
# rmse = np.average(final**2)**0.5
# print("weight : ",weight)
# print("avg : ",np.average(final),"rmse : ",rmse)

######################
# 	read testfile    #
######################
testfile = pd.read_csv("./test.csv",encoding='big5')
cnt_row = 1
testData = []
for index,row in testfile.iterrows():
	if cnt_row%18 == 7:
		testData.append([1])
		for i in range(2,11):
				testData[cnt_row//18].append(float(row[i]))
				testData[cnt_row//18].append(float(row[i])**2)
	if cnt_row%18 == 8:
		for i in range(2,11):
				testData[cnt_row//18].append(float(row[i]))
				testData[cnt_row//18].append(float(row[i])**2)
	if cnt_row%18 == 9:
		for i in range(2,11):
				testData[cnt_row//18].append(float(row[i]))
				testData[cnt_row//18].append(float(row[i])**2)
	cnt_row +=1
print("Data[0] : ",testData[0])
print("Data length : ",len(testData))

##########################
# 	calculate predict    #
##########################
nptestdata = np.array(testData)
output = np.dot(nptestdata,weight)
print("output len : ",len(output))

#########################
# 	write outputfile    #
# #########################
fw = open("./output.csv","w")
csvCursor = csv.writer(fw)
writefile = ["id","value"]
csvCursor.writerow(writefile)
for i in range(240):
	strtest= "id_"+str(i)
	writefile[0] = strtest
	writefile[1] =output[i]
	csvCursor.writerow(writefile)