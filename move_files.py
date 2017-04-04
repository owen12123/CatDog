import csv
import os

def csv_reader(file_obj):
    labels = []
    reader = csv.reader(file_obj)
    for row in reader:
        labels.append(row)
    return labels

face_path = 'C:\\Users\\Raymond\\Desktop\\faces\\'

path = 'Y_Train.csv'
with open(path, "r") as file:
    labs = csv_reader(file)

del labs[0]

for i in range(0,len(labs)):
    curr_path = face_path + labs[i][0]
    if i == 1:
        print(curr_path)
    
    if os.path.isfile(curr_path):
        if labs[i][1] == '1':
            os.rename(curr_path, 'C:\\Users\\Raymond\\Desktop\\dogs\\'+labs[i][0])
        else: 
            os.rename(curr_path, 'C:\\Users\\Raymond\\Desktop\\cats\\'+labs[i][0])