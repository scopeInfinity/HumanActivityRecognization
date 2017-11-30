import numpy as np
import cv2
import os

def encodeFeature(in_):
	# If we want to reduce dimension or similar
	return in_

def getTime(frames,tsecs,_id):
	time = _id*tsecs/frames
	ms = str(time).split(".")[1]
	s = int(str(time).split(".")[0])
	m,s = divmod(s,60)
	h,m = divmod(m,60)
	time = "%0d:%02d:%02d.%s" % (h,m,s,ms)
	return time

def getFrameCount(fname):
	return int(os.popen("ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1  %s"%fname).read())

def load_video(fname,startf,endframe):
	duration=os.popen("ffmpeg -i %s 2>&1 | grep Duration | awk '{print $2}' | tr -d ,"%fname).read().split(":")
	tsecs = 0
	print "Loading video %s " % fname
	# print duration
	tsecs += int(duration[0])*3600
	tsecs += int(duration[1])*60
	tsecs += float(duration[2])
	# print tsecs
	frames = getFrameCount(fname)
	# print frames
	os.popen("rm -r temp")
	os.popen("mkdir temp")
	stime = getTime(frames,tsecs, startf)
	etime = getTime(frames,tsecs, endframe)
	os.system(("ffmpeg -i %s -vsync vfr temp/"%fname+"%d.jpg -hide_banner"))
	
	print "All frames extracted! %s " % fname
	os.system("sleep 5s")
	
def bestchoice(a,b):
	# minx,miny,maxx,maxy
	return (min(a[0],b[0]),min(a[01],b[1]),max(a[2],b[2]),max(a[3],b[3]))
INF = 100000000
def dfs(img,r,c):
	d = (INF,INF,-INF,-INF)
	if r<0 or c<0 or r>=len(img) or c>=len(img[0]):
		return d
	if img[r][c]>0:
		img[r][c]=0
		d = bestchoice(d,(c,r,c,r))
		d = bestchoice(d,dfs(img,r-1,c))
		d = bestchoice(d,dfs(img,r-1,c-1))
		d = bestchoice(d,dfs(img,r-1,c+1))
		d = bestchoice(d,dfs(img,r+1,c+1))
		d = bestchoice(d,dfs(img,r+1,c-1))
		d = bestchoice(d,dfs(img,r+1,c))
		d = bestchoice(d,dfs(img,r,c-1))
		d = bestchoice(d,dfs(img,r,c+1))
	return d


def getFeatures(fname,startf,endframe):
	load_video(fname,startf,endframe)
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	fFrame = None
	clipFeature = []
	for i2,f in enumerate(range(endframe- startf)):
		try:
			i= i2+startf
			fn = 'temp/%d.jpg'%i
			# print "loading image %s" % fn
			img = cv2.imread(fn)
			img = cv2.GaussianBlur(img, (5, 5), 0)

			frame = img
			fgmask = fgbg.apply(img)
			thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
			mx =0
			my =0
			conter = 0
			for i,row in enumerate(thresh):
				for j,cell in enumerate(row):
					if cell >0:
						mx += j
						my += i
						conter+=1
			if conter ==0 :
				continue
			my /= conter
			mx /= conter
			if conter>0.5*len(thresh[0])*len(thresh):
				continue

			(minx,miny,maxx,maxy) = dfs(thresh.copy(),int(my),int(mx))
			if minx == INF or miny == INF or maxx == -INF or maxy == -INF:
				continue

			newimg = thresh[miny:maxy, minx:maxx]
			height = maxy-miny
			width = maxx-minx
			

			K = 4
			if height<=K or width<=K:
				continue
			ox = np.zeros(K)
			oy = np.zeros(K)
			# Row Wise
			for i in range(K):
				blockH = height/K
				aa = newimg[i*blockH:(i+1)*blockH, : ]
				# print aa
				oy[i] = np.mean(aa)
			# Column Wise
			for i in range(K):
				blockW = width/K
				ox[i] = np.mean(newimg[:, i*blockW:(i+1)*blockW])
			ox/=255.0
			oy/=255.0


			motionFeature = np.append(ox,oy)
			# print "Motion Feature"
			# print motionFeature
			encoded = encodeFeature(motionFeature)
			clipFeature.append(motionFeature)


		except Exception as e:
			print(e)
	return clipFeature

