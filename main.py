import feature
from hmmlearn import hmm
import pickle
from sklearn.externals import joblib
import numpy as np
import glob,random
import sys
import argparse

allowedclass = ["running","handclapping","jogging","walking"]#,"boxing","handwaving"

# #data set
# # http://www.nada.kth.se/cvap/actions/

def train():
	print "Train Started"
	with open("00sequences.txt",'r') as f:
		lines = f.readlines()

	fclass = dict()
	clength = dict()
	model = dict()
	countdata = dict()
	for c in allowedclass:
		fclass[c]= []
		clength[c] = []
		countdata[c] = 0
		model[c] = hmm.GaussianHMM(10, "full")

	for line in lines:
		tok =  line.split()
		if len(tok)>0:
			# print tok
			fname = "videos/" + tok[0] + "_uncomp.avi"
			c = tok[0].split("_")[1]
			if c not in allowedclass:
				continue

			print "Fname : %s \t Class : %s " % (fname,c)
			if countdata[c]>=1:
				continue
			countdata[c] += 1
			frames = tok[2:]
			for f in frames:
				sf_ = f.split("-")
				sf = int(sf_[0])
				if sf_[1][-1]==',':
					sf_[1]=sf_[1][0:-1]
				ef = int(sf_[1])
				#print sf,ef
				f = feature.getFeatures(fname,sf,ef)
				clength[c].append(len(f))
				print f
				fclass[c].extend(f)
	for c in allowedclass:
		print "Learning for '%s'" % c
		print fclass[c]
		print clength[c]
		model[c].fit(np.array(fclass[c]),np.array(clength[c]))
		fn = "model/model_"+c+".pkl"
		joblib.dump(model[c], fn)


test_log_total   = 0
test_log_correct = 0
glob_msg = []

def addMessage(s):
	glob_msg.append(s)
	print s

def test(name):
	global test_log_total
	global test_log_correct
	print "Started Test"
	model = dict()
	prob = dict()
	for c in allowedclass:
		fn = "model/model_"+c+".pkl"
		prob[c] = []
		model[c] = joblib.load(fn)
	fname = name# "videos/" + name + "_uncomp.avi"
	print fname
	totalframe = feature.getFrameCount(fname)
	W = 10
	if totalframe<W:
		if totalframe<2:
			print "Too less features frame, ignoring..."
			return
		else:
			W=totalframe
	f = feature.getFeatures(fname,0,totalframe)
	totalframe = len(f)
	
	print "Feature Frames %d" % len(f)
	for c in allowedclass:
		print "Test for '%s'" % c
		for i in range(0,totalframe/W):
			sf = i*W
			ef = (i+1)*W
			subfeature = f[sf:ef]
			print len(subfeature)
			p = model[c].score(np.array(subfeature))
			prob[c].append(p)
	probClass = dict()
	bClass = "none"
	bProb = -float('inf')
	for c in allowedclass:
		probClass[c] = np.sum(prob[c])
		if bProb<probClass[c]:
			bProb=probClass[c]
			bClass = c
	aclass = name.split("_")[1]
	addMessage("File : %s\tPredicted : [%s class] \tActual : [%s class]" % (name,bClass,aclass))
	addMessage(str(probClass))
	test_log_total  +=1
	if aclass == bClass:
		test_log_correct+=1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("mode",choices=["train","test"])
	args = parser.parse_args()
	if args.mode == "train":
		train()
	else:
		fnames = []
		testCountPerClass = 1
		for c in allowedclass:
			X=glob.glob("videos/person*_%s_*"%c)
			random.shuffle(X)
			fnames.extend(X[:testCountPerClass])
		print "Testing for ...."
		print fnames
		for fn in fnames:
			test(fn)
		for l in glob_msg:
			print l
		print ""
		print "*" * 20
		print "Test Accuracy : %2.2f " % (test_log_correct*100.0/test_log_total)
