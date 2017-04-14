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
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to the input image")

ap.add_argument("-c", "--cascade",
	default="Classifier_All/allcatdogbody.xml",
	help="path to face detector haar cascade")
args = vars(ap.parse_args())

yPath = 'Y_Train.csv'
with open(yPath, "r") as file:
    csv = csv_reader(file)
del csv[0]

<<<<<<< HEAD
filenames = []
for root, dirs, files in os.walk('C:/Users/Raymond/Desktop/trainSet'):
    filenames = files 

xData = np.zeros((len(filenames),256), dtype=np.float64)
#yData = np.zeros((len(filenames)), dtype=object)
yData = []
xPath = 'C:/Users/Raymond/Desktop/trainSet/'

#yData = np.asarray(csv)
#yData = np.delete(yData,0,1)
#yData = yData.flatten()

zero_rows = 0

for i in range(0,len(filenames)):
	curr_path = xPath + filenames[i]

	# load the input image and convert it to grayscale
	image = cv2.imread(curr_path)
	image = cv2.resize(image, (500, image.shape[0]*500//image.shape[1]))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)		
	# load the face detector Haar cascade, then detect faces
	# in the input image
	detector = cv2.CascadeClassifier(args["cascade"])
	rects = detector.detectMultiScale(gray, scaleFactor=1.3,
		minNeighbors=10, minSize=(75, 75))		
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
	lbp = local_binary_pattern(cropped_img, 8, 1, "default")
	hist, _ = np.histogram(lbp, 256, density=True)
	xData[i] = hist

	zero = not hist.any()
	if(zero):
=======
xData = np.zeros((len(csv),256), dtype=np.float64)

xPath = 'C:/Users/Raymond/Documents/opencv Shit/X_Train/'
#xPath = ''

yData = np.asarray(csv)
yData = np.delete(yData,0,1)
yData = yData.flatten()

for i in range(0,len(csv)):
	curr_path = xPath + csv[i][0]
	if os.path.isfile(curr_path):

		# load the input image and convert it to grayscale
		image = cv2.imread(curr_path)
		image = cv2.resize(image, (500, image.shape[0]*500//image.shape[1]))
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# load the face detector Haar cascade, then detect faces
		# in the input image
		detector = cv2.CascadeClassifier(args["cascade"])
		rects = detector.detectMultiScale(gray, scaleFactor=1.3,
			minNeighbors=10, minSize=(75, 75))
		
>>>>>>> origin/master
		print(curr_path)

	img_index = int(filenames[i][:-4])
	yData.append(csv[img_index][1])

# Remove zero-rows from xData array     
#xData = xData[~np.all(xData == 0, axis=1)] 

clf = svm.LinearSVC()

#print(xData)
#print(yData)

clf.fit(xData,yData)

joblib.dump(clf, 'equalcatdog.pkl')