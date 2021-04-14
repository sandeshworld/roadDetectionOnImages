
#look at splitImages instead -> this file still requires us to open and save too many images

#Sandesh
import os
import numpy as np
from PIL import Image

def LoadSceneAndLabel(dirnameScene, dirnameLabel, numImages):
    count = 0
    sceneX = []
    sceneY = []
    for sceneName in os.listdir(dirnameScene):
        try:
            scene = Image.open(os.path.join(dirnameScene, sceneName))
            sceneLabel = Image.open(os.path.join(dirnameLabel, sceneName[:-4]+"_drivable_color.png"))
        except:
            print(count)

        #had issue with PIL saying too many things open
        scene1 = scene.copy()
        sceneLabel1 = sceneLabel.copy()
        # if count == 0:
        #     scene.show()
        #     sceneLabel.show()
        #print(scene.size)
        #print(sceneLabel.size)
        sceneX.append(scene1)
        sceneY.append(sceneLabel1)


        scene.close()
        sceneLabel.close()

        count += 1
        if count >= numImages:
          break
    return sceneX, sceneY

dataX, dataY = LoadSceneAndLabel(dirnameScene="bdd100k_images/images/100k/train", dirnameLabel="bdd100k_drivable_map/drivable_maps/color_labels/train", numImages=30000)
valX, valY = LoadSceneAndLabel(dirnameScene="bdd100k_images/images/100k/val", dirnameLabel="bdd100k_drivable_map/drivable_maps/color_labels/val", numImages=1100)

import math

def compress(x, y, rate=0.5):
    global xSize
    global ySize
    print((x[0].size))
    xSize = math.floor(((x[0].size)[0])*rate)
    ySize = math.floor(((y[0].size)[1])*rate)
#     print(xSize)
#     print(ySize)
    sceneX = [img.resize([xSize, ySize]) for img in x]  
    sceneY = [img.resize([xSize, ySize]) for img in y]  
    #sceneX[0].show()
    #sceneY[0].show()
    return sceneX, sceneY

# #print(SceneLabelPair[0][0].size)
dataX, dataY = compress(dataX, dataY, rate=0.1)
# print(SceneLabelPair[0][0].size)
valX, valY = compress(valX, valY, rate=0.1)

import torchvision.transforms.functional as TF

import random

#list is pass by reference so copy isn't made

def transformImages(image, mask):

    for n in range(len(image)):
#         # Resize
#         resize = transforms.Resize(size=(520, 520))
#         image = resize(image)
#         mask = resize(mask)

#         # Random crop
#         i, j, h, w = transforms.RandomCrop.get_params(
#             image, output_size=(512, 512))
#         image = TF.crop(image, i, j, h, w)
#         mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image[n] = TF.hflip(image[n])
            mask[n] = TF.hflip(mask[n])

        # Random vertical flipping
        if random.random() > 0.5:
            image[n] = TF.vflip(image[n])
            mask[n] = TF.vflip(mask[n])

    
#transform dataX and dataY, doesn't return anything since passed by reference
# transformImages(dataX, dataY)

#convert non zero label to be the same number so it's not different based on different lane categorization

def convert_to_array(x, y):
    sceneX = []
    sceneY = []
    count = 0
    for img in x:
        #img = img.convert('LA')  # convert to grayscale
        #img = np.squeeze(np.array(img)[:,:,:])
        img = np.array(img)
        #divide by 255.0
        img = np.divide(img, 255.0)
        sceneX.append(img)
    for img in y:
        img = img.convert('LA')  # convert to grayscale
        # if count == 0:
        #     img.show()
        count = 1
        img = np.squeeze(np.array(img)[:,:,0])
        img = np.where(img > 0, 1, 0)
        sceneY.append(img)
#     img = np.squeeze(np.array(img)[:, :, 0]) #squeeze to remove the unnecessary dimension after conversion to grayscale
#     label_img = np.squeeze(np.array(label_img)[:, :, 0])
    return np.array(sceneX), np.array(sceneY)

dataX, dataY = convert_to_array(dataX, dataY)
valX, valY = convert_to_array(valX, valY)


from sklearn.model_selection import train_test_split

split = 2/7.0
print(split)

trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=split) 

train_num = trainX.shape[0]
test_num = testX.shape[0]
x_dim = trainX.shape[2]
y_dim = trainX.shape[1]
val_num = valX.shape[0]


import torch
# train

trainX = trainX.reshape(train_num, 3, y_dim, x_dim).astype(float)
#print(trainX[0])
#print(trainX.shape)
#trainX  = torch.from_numpy(trainX).type(torch.FloatTensor)

# trainY = trainY.reshape(train_num, 1, x_dim, y_dim).astype(float)
# trainY  = torch.from_numpy(trainY)
# print(trainY.shape)

trainY = trainY.reshape(train_num, 1, y_dim, x_dim).astype(float)
#trainY  = torch.from_numpy(trainY).type(torch.LongTensor)
#trainY  = torch.from_numpy(trainY).type(torch.FloatTensor)
#print(trainY.shape)



# val
valX = valX.reshape(val_num, 3, y_dim, x_dim).astype(float)
#print(valX[0])
#print(valX.shape)
#valX  = torch.from_numpy(valX).type(torch.FloatTensor)

# valY = valY.reshape(val_num, 1, x_dim, y_dim).astype(float)
# valY  = torch.from_numpy(valY)


valY = valY.reshape(val_num, 1, y_dim, x_dim).astype(float)
#valY  = torch.from_numpy(valY).type(torch.LongTensor)
#valY  = torch.from_numpy(valY).type(torch.FloatTensor)
# //TODO Test


# val
testX = testX.reshape(test_num, 3, y_dim, x_dim).astype(float)
#print(testX[0])
#print(testX.shape)
#valX  = torch.from_numpy(valX).type(torch.FloatTensor)

# valY = valY.reshape(val_num, 1, x_dim, y_dim).astype(float)
# valY  = torch.from_numpy(valY)


testY = testY.reshape(test_num, 1, y_dim, x_dim).astype(float)
#valY  = torch.from_numpy(valY).type(torch.LongTensor)
#valY  = torch.from_numpy(valY).type(torch.FloatTensor)
# //TODO Test

import os

# save files now into the directory
files = [[trainX, trainY], [valX, valY], [testX, testY]]
fileNames = [["trainX", "trainY"], ["valX", "valY"], ["testX", "testY"]]
csvNames = ["train.csv","val.csv", "test.csv"]

csvs = [[],[],[]]


for i in range(len(files)):
    imNum = 0
    for arr in range(len(files[i][0])):
        fNameX = fileNames[i][0] + "/" + "imX" + str(imNum)
        fNameY = fileNames[i][1] + "/" + "imY" + str(imNum)

        try:
            np.save(fNameX, files[i][0][arr], allow_pickle=True, fix_imports=True)
        except:
            os.mkdir(fileNames[i][0])
            np.save(fNameX, files[i][0][arr], allow_pickle=True, fix_imports=True)

        try:
            np.save(fNameY, files[i][1][arr], allow_pickle=True, fix_imports=True)
        except:
            os.mkdir(fileNames[i][1])
            np.save(fNameY, files[i][1][arr], allow_pickle=True, fix_imports=True)


        csvs[i].append(["imX" + str(imNum)+".npy", "imY" + str(imNum)+".npy"])

        imNum += 1
    
    import csv

    with open(csvNames[i], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csvs[i])


#print(csvs)
