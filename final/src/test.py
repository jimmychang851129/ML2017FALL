import jieba
from gensim.models import Word2Vec
from gensim import corpora
from gensim import models
from gensim.models import word2vec
from gensim.models import word2vec
from hanziconv import HanziConv
import numpy as np
import io
from sklearn.metrics.pairwise import cosine_similarity 
import sys

testfile = sys.argv[1]
outputcsv = sys.argv[2]
modelfile = "./bestmodel.bin"
wordvec_size = 176
word_limit = 0

def parseData(data):
	return [(sent.replace('A:','')).replace('B:','').replace('C:','').strip('\n') for sent in data]

def jiebaSeg(lines):
	segLine = []
	words = jieba.cut(lines)
	for word in words:
		if word != ' ' and word != '':
			segLine.append(word)
	return segLine


def sim(quest,choice):
	alpha = 1e-2
	choice_len = 6
	questSeg = []
	optSeg = []
	for q in quest:
		qSeg = jiebaSeg(q)
		questSeg += [ i for i in qSeg]
	for o in choice:
		oSeg = jiebaSeg(o)
		if oSeg is None:
			continue
		optSeg.append(oSeg)
	sum_sim = np.zeros((choice_len,1),dtype=float)
	ques_vec = np.zeros((1, wordvec_size))
	q_count = 0
	for word in questSeg:
		try:
			prob = model.wv.vocab[word].count / total_count
			ques_vec += model[word]*(alpha/(alpha+prob))
			q_count += 1
		except Exception as e:
			# do nothing
			q_count = q_count
	if q_count != 0:
		ques_vec /= q_count
	opt_vec = np.zeros((choice_len, 1, wordvec_size))
	for i in range(choice_len):
		o_count = 0
		for word in optSeg[i]:
			try:
				prob = model.wv.vocab[word].count / total_count
				opt_vec[i] += model[word]*(alpha/(alpha+prob))
				o_count += 1
			except Exception as e:
				# do nothing
				o_count = o_count
		if o_count != 0:
			opt_vec[i] /= o_count
	
	for i in range(choice_len):
		sum_sim[i] += abs(cosine_similarity(ques_vec,opt_vec[i])[0][0])
	return sum_sim

model =Word2Vec.load(modelfile)
vocab = list(model.wv.vocab)
total_count = 0
for tmpv in vocab:
	total_count += model.wv.vocab[tmpv].count

print("read testfile")
question = []
choice = []
cnt = 0
with io.open(testfile,'r',encoding='utf-8') as f:
	f.readline()
	for line in f:
			line = line.split(',')
			tmp_question = parseData(HanziConv.toSimplified(line[1]).split('\t'))
			question.append(tmp_question)
			tmp_choice = parseData(HanziConv.toSimplified(line[2]).split('\t'))
			choice.append(tmp_choice)
			
print("start sim")
M = []
for i,q in enumerate(question):
	mat = sim(q,choice[i])
	M.append(mat.argmax())


fw = open(outputcsv,'w')
fw.write('id,ans\n')
for i in range(len(M)):
    fw.write('{},{}\n'.format(i+1,M[i]))

print("finish writing output")
