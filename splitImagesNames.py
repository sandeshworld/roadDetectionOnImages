# Sandesh
import os
import csv
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def LoadSceneAndLabelName(dirnameScene, numImages):
    count = 0
    sceneXname = []
    sceneYname = []
    for sceneName in os.listdir(dirnameScene):
        sceneXname.append(sceneName)
        sceneYname.append(sceneName[:-4] + "_drivable_color.png")
        count += 1
        if count >= numImages:
            break
    return sceneXname, sceneYname


dataX_names, dataY_names = LoadSceneAndLabelName(dirnameScene="bdd100k_images/images/100k/train", numImages=70000)
valX_names, valY_names = LoadSceneAndLabelName(dirnameScene="bdd100k_images/images/100k/val", numImages=20000)


split = 2 / 7.0

trainX_names, testX_names, trainY_names, testY_names = train_test_split(dataX_names, dataY_names, test_size=split)

# save files now into the directory
files = [[trainX_names, trainY_names], [valX_names, valY_names], [testX_names, testY_names]]
csvNames = ["train.csv", "val.csv", "test.csv"]

for i in range(3):
    with open(csvNames[i], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nameX", "nameY"])  # title for the columns

        for j in range(len(files[i][0])):
            nameX = files[i][0][j]
            nameY = files[i][1][j]
            writer.writerow([nameX, nameY])



