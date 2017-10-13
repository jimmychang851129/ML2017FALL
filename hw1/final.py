import pandas as pd
import numpy as np
import math
import csv
import sys

weight =  [  4.08497417e-02 ,-1.29559146e-02,  -5.22090139e-05,  -2.97214402e-03,
   1.12859303e-04,  -1.74243831e-03,   5.10803446e-05,   2.27135809e-04,
  -8.11219606e-05,  -8.15412888e-03,   9.45108574e-05,   1.44203736e-03,
  -2.44774970e-04,  -1.68846823e-02,   1.65553423e-04,  -5.34689924e-04,
  -1.16708385e-04,   7.04977512e-02,  -5.07295370e-05,  -7.06588208e-03,
   8.34136489e-05,  -7.11111357e-03,  -1.03957140e-05,  -3.03019657e-03,
   7.13808096e-05,   1.05598322e-02,   1.00720125e-04,   8.94468381e-03,
  -4.91103896e-04,  -1.26479659e-02,   2.27527303e-04,  -1.23062148e-02,
   2.66422832e-04,   2.94463993e-02,  -6.20611470e-04,   8.88067181e-02,
   1.35191304e-04,   2.20453329e-02,  -7.91527994e-04,   2.04918253e-03,
   1.76305236e-04,   1.91439136e-02,   1.66762170e-03,  -1.17674134e-02,
  -2.22820867e-03,   2.45060119e-02,  -4.71003989e-05,   6.01427696e-02,
   4.40365372e-03,  -8.27677607e-02,  -5.79083859e-03,   1.64188874e-01,
  -1.38979251e-03,   5.68003115e-01,   4.97342698e-03]

###############
#   config    #
###############
print("argv : ",sys.argv)
inputfile = sys.argv[1]
outputfile = sys.argv[2]

######################
#   read testfile    #
######################
testfile = pd.read_csv(inputfile,encoding='big5')
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
#   calculate predict    #
##########################
nptestdata = np.array(testData)
output = np.dot(nptestdata,weight)
print("output len : ",len(output))

#########################
#   write outputfile    #
# #########################
fw = open(outputfile,"w")
csvCursor = csv.writer(fw)
writefile = ["id","value"]
csvCursor.writerow(writefile)
for i in range(240):
  strtest= "id_"+str(i)
  writefile[0] = strtest
  writefile[1] =output[i]
  csvCursor.writerow(writefile)