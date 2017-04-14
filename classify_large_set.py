import os
import csv

def csv_writer(data, path):

#    Write data to a CSV file path
    
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

dataPath = 'C:/Users/Raymond/Downloads/train'
#dataPath = 'C:/Users/Raymond/Documents/CatDog/testphotos'

outputpath = 'C:/Users/Raymond/Documents/CatDog/hugeTrain.csv'

filenames = []
clas = ''
csv_out = []

csv_out.append("Image,Label".split(","))

for root, dirs, files in os.walk(dataPath):
    filenames = files 

print(filenames[1379][4:-4])

'''
for i in range(0,len(filenames)):
    if filenames[i][:3] == 'dog':
    	clas = '1'
    else:
    	clas = '0'
    csv_out.append((filenames[i]+","+clas).split(","))

csv_writer(csv_out,outputpath)
'''