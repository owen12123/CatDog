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

xData = np.zeros((len(filenames),256*25), dtype=np.float64)
#yData = np.zeros((len(filenames)), dtype=object)
yData = []
xPath = samplePath + '/'

#yData = np.asarray(csv)
#yData = np.delete(yData,0,1)
#yData = yData.flatten()

#zero_rows = 0

for i in range(0,len(filenames)):
	curr_path = xPath + filenames[i]

	# load the input image and convert it to grayscale
	image = cv2.imread(curr_path)
	image = cv2.resize(image, (150, image.shape[0]*150//image.shape[1]))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.blur(gray,(5,5))
	# load the face detector Haar cascade, then detect faces
	# in the input image
	rects1 = detector1.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))
	rects2 = detector2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))	

	rects = []
	if len(rects1)>0 and len(rects2)==0:
		for j in range(0,len(rects1)):
			rects.append(rects1[j])
	elif len(rects2)>0 and len(rects1)==0:	
		for j in range(0,len(rects2)):
			rects.append(rects2[j])
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
	cell_height = yhMax//5
	cell_width = xwMax//5

	cells = []
	cell_r,cell_l,cell_u,cell_d, = 0,0,0,0

	for j in range(0,5):
		for k in range(0,5):
			cell_l = cell_width*j
			cell_r = cell_width*(j+1)
			cell_u = cell_height*k
			cell_d = cell_height*(k+1)
			if j == 4:
				cell_r = xwMax
			if k == 4:
				cell_d = yhMax
			img_part = cropped_img[cell_l:cell_r,cell_u:cell_d]
			cells.append(img_part)
	print(cells[0])
	for j in range(0,25):
		lbp = local_binary_pattern(cells[j], 8, 1, "default")
		hist, _ = np.histogram(lbp, 256, density=True)
		xData[i,(25*j):(25*j+256)] = hist

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

joblib.dump(clf, 'lbpCells.pkl')