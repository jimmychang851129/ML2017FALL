import sys
f = open(sys.argv[1],'r')
output = open("Q1.txt","w")

a = f.read()
word_list = a.split()
dict = {}
list = []

for word in word_list:
	if(word in dict):
		dict[word] += 1
	else:
		dict[word] = 1
		list.append(word)

length = len(list)
for idx, val in enumerate(list):
	if(idx != length):
		#print(val,idx,dict[val])
		strtmp =val+" "+str(idx)+" "+str(dict[val])+"\n"
		output.write(strtmp)
	else:
		strtmp =val+" "+str(idx)+" "+str(dict[val])
		output.write(strtmp)
