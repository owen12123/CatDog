import csv
import argparse
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
from sklearn import svm
from sklearn.externals import joblib

def csv_reader(file_obj):
    labels = []
    reader = csv.reader(file_obj)
    for row in reader:
        labels.append(row)
    return labels

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to the input image")

#ap.add_argument("-c", "--cascade",
#	default="Classifier_All/allcatdogbody.xml",
#	help="path to face detector haar cascade")
#args = vars(ap.parse_args())

xml1 = 'Classifiers/4kdogcascade.xml'
xml2 = 'Classifier_cat_4k/cascade.xml'

detector1 = cv2.CascadeClassifier(xml1)
detector2 = cv2.CascadeClassifier(xml2)

sift = cv2.xfeatures2d.SIFT_create()

yPath = 'Y_Train.csv'
#yPath = 'hugeTrain2.csv'
with open(yPath, "r") as file:
    csv = csv_reader(file)
del csv[0]

#samplePath = 'C:/Users/Raymond/Downloads/train'
samplePath = 'C:/Users/Raymond/Desktop/trainSet'

filenames = []
for root, dirs, files in os.walk(samplePath):
    filenames = files 

#xData = np.zeros((len(filenames),256), dtype=np.float64)
yData = []
xPath = samplePath + '/'

# Create array of descriptors with room for 30 descriptors per image
xData = np.zeros((len(filenames),25,128))    

for i in range(0,len(filenames)):
	curr_path = xPath + filenames[i]
	#print(curr_path)
	# load the input image and convert it to grayscale
	image = cv2.imread(curr_path)
	#image = cv2.resize(image, (150, image.shape[0]*150//image.shape[1]))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#gray = cv2.blur(gray,(5,5))
	# load the face detector Haar cascade, then detect faces
	# in the input image
	rects1 = detector1.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))
	rects2 = detector2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))	

	rects = []
	if len(rects1)>0 and len(rects2)==0:
		for i in range(0,len(rects1)):
			rects.append(rects1[i])
	elif len(rects2)>0 and len(rects1)==0:	
		for i in range(0,len(rects2)):
			rects.append(rects2[i])
	elif len(rects1)>0 and len(rects2)>0 : 
		rects.append(rects1[0])
		rects.append(rects2[0])

	#print(curr_path)		
	X = []
	Y = []
	XW = []
	YH = []		
	# loop over the reactnalges and record edge positions
	for (i, (x, y, w, h)) in enumerate(rects):
		X.append(x)
		Y.append(y)
		XW.append(x+w)
		YH.append(y+h)		
	# get minimum and maximum rectangle coordinates
	if len(X) != 0:
		xMin = min(X)
		yMin = min(Y)
		xwMax = max(XW)
		yhMax = max(YH)
	else:
		height,width = gray.shape
		xMin,yMin,xwMax,yhMax = 0,0,width,height		
	# draw combined rectangle
	#cv2.rectangle(image, (xMin, yMin), (xwMax, yhMax), (0, 0, 255), 2)		
	# get portion of image within large rectangle
	cropped_img = gray[yMin:yhMax,xMin:xwMax]		
	# compute local binary pattern of cropped image and its normalized histogram
	
   	# Detext all keypoints
	kp = sift.detect(cropped_img,None)

	if len(kp) > 25:
       	# Sort keypoints based on response and keep only the top 100
		kp = sorted(kp, key=lambda keyp:keyp.response, reverse=True)
		bestKp = kp[0:25]
	else:
		bestKp = kp
	
	if len(kp) > 0: 

   		# Use the top 100 keypoints to calculate descriptors
		bestKp, desc = sift.compute(cropped_img,bestKp)
	
		if len(desc) < 25:
			xData[i,0:len(desc)] = desc
			xData[i,len(desc):25] = None
		else:
			xData[i,0:25] = desc
		
		# Remove zero-rows from descriptor array     
		#descriptors = descriptors[~np.all(descriptors == 0, axis=1)] 
	
		#lbp = local_binary_pattern(cropped_img, 8, 1, "default")
		#hist, _ = np.histogram(lbp, 256, density=True)
		#xData[i] = hist
	
		#zero = not hist.any()
		#if(zero):
		#print(filenames[i])
		'''
		if filenames[i][:3] == 'dog':
			img_index = int(filenames[i][4:-4])+12500
		else:
			img_index = int(filenames[i][4:-4])
		'''
	
		img_index = int(filenames[i][:-4])
	
		yData.append(csv[img_index][1])

# Remove zero-rows from xData array     
#xData = xData[~np.all(xData == 0, axis=1)] 

clf = svm.LinearSVC()

#print(xData)
#print(yData)

clf.fit(xData,yData)

joblib.dump(clf, 'SIFT.pkl')