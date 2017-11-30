import feature
from hmmlearn import hmm
import pickle
from sklearn.externals import joblib
import numpy as np
allowedclass = ["running","handclapping"]#,"jogging","boxing","handwaving","walking"]

def train():
	print "Started"
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
		model[c] = hmm.GaussianHMM(2, "full")
		# x1 = np.asarray([[0.1,0.2],[0.3,0.6],[0.5,0.9]],dtype=np.float32)
		# x2 = np.asarray([[0.23,0.2],[0.23,0.6],[0.53,0.39]],dtype=np.float32)
		# inn = np.concatenate([x1,x2])
		# ll = np.array([len(x1),len(x2)])
		# model[c].fit(inn, ll)
		# exit()

	for line in lines:
		tok =  line.split()
		if len(tok)>0:
			# print tok
			fname = "videos/" + tok[0] + "_uncomp.avi"
			print fname
			c = tok[0].split("_")[1]
			if c not in allowedclass:
				continue
			print c
			frames = tok[2:]
			for f in frames:
				if countdata[c]>1:
					continue
				countdata[c] += 1
				sf_ = f.split("-")
				# print sf_
				sf = int(sf_[0])
				if sf_[1][-1]==',':
					sf_[1]=sf_[1][0:-1]
				# print sf_[1]
				ef = int(sf_[1])
				print sf,ef
				f = feature.getFeatures(fname,sf,ef)
				clength[c].append(len(f))
				print f
				fclass[c].extend(f)
	for c in allowedclass:
		print "Learning for '%s'" % c
		print fclass[c]
		print clength[c]
		model[c].fit(np.array(fclass[c]),np.array(clength[c]))
		fn = "model_"+c+".pkl"
		joblib.dump(model[c], fn)
		# with open(fn,'w') as f:
		# 	pickle.dump(model[c],f)

def test(name):
	print "Started Test"
	model = dict()
	prob = dict()
	for c in allowedclass:
		fn = "model_"+c+".pkl"
		prob[c] = []
		model[c] = joblib.load(fn)
	fname = "videos/" + name + "_uncomp.avi"
	totalframe = 100#feature.getFrameCount(fname)
	W = 10

	f = feature.getFeatures(fname,0,totalframe)
	totalframe = len(f)
	print ">>>> %d" % len(f)
	for c in allowedclass:
		print "Test for '%s'" % c
		for i in range(0,totalframe/W):
			sf = i*W
			ef = (i+1)*W
			subfeature = f[sf:ef]
			print len(subfeature)
			p = model[c].score(np.array(subfeature))
			prob[c].append(p)
	print prob

if __name__ == '__main__':
	#train()
	test('person16_handclapping_d4')