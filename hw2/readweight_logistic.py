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
bias = -2.10650124564
cnt = 0
index_nor = [0,1,3,4,5]

#############
#	init    #
#############
weight = [0.349467244752,0.0746457849123,0.404772103332,2.37745706161,0.260773016374,0.367775838949,0.102423755305,-0.0190053906425,-0.0823182392583,0.0484035205459,0.0513260735324,-0.103960566121,-0.0405614633437,-0.155159729459,-0.0864369337947,-0.222035549923,-0.236309808536,-0.0975010039408,-0.132335035926,-0.159848636593,-0.250769367962,-0.188820770191,-0.000266608269786,0.00300618312969,0.221873042835,0.181058168243,-0.244803051115,0.2158125202,-0.594850547281,0.188572207209,-0.0740735823761,-0.340628763407,0.0454288421275,0.583875464588,-0.113257732189,-0.693873643906,-0.19639149721,-0.150282940018,-0.0464461440589,-0.0230073315896,-0.0244187194493,0.211680038254,-0.19559641581,-0.166145129468,-0.103031919038,-0.291645898334,-0.296593961728,0.122802540452,0.0605086809978,0.0428658312638,0.0864161090333,-0.0547114588677,-0.0865413309212,-0.0367054501166,0.19871275742,-0.0804442333323,-0.276225772794,0.0999592183738,0.274052663578,-0.0474333583689,0.0309714595665,-0.0272286410411,-0.0283529529427,0.0303700874872,0.0324731442722,0.023903384949,-0.0315358335456,-0.0854954429182,0.0216844280677,-0.0808681011444,-0.00659529512479,-0.0307299285876,0.0188745723535,0.0180690528574,0.0320461892892,-0.0291876456859,-0.00794538394066,0.000459765574276,-0.024126740299,-0.0243719367949,-0.000941467182564,-0.00102706349439,-0.0173790322683,0.00376782076579,0.0163863531821,0.0402357440476,0.0048182343125,0.0197484910675,-0.0118392782966,-0.0673310677297,-0.0234133495505,-0.147491424899,-0.0237324961221,0.0376739037851,0.0024119064028,0.00108697175484,-0.016713188765,0.00136243287783,-0.0487519554019,0.00362607394985,-0.0124674683387,-0.00757714998961,0.0778851694316,-0.0454343021907,0.0161945406546,-0.0163849493971]
weight = np.array(weight)
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