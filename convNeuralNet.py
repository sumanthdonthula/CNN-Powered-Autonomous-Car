import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
import argparse
import os
import cv2
import numpy as np
import matplotlib.image as mpimg

#Image dimensions
imageHeight, imageWidth, imageChannels = 160, 320, 3
inputShape = (imageHeight, imageWidth, imageChannels)

#Load Image from path
def loadImage(dataDir, imageFile):
    return mpimg.imread(os.path.join(dataDir, imageFile.strip()))

#Crop Image to focus on area of interest
def cropImage(image):
    return image[50:-25, :, :]

#Resize image according to NN
def resizeImage(image):
    return cv2.resize(image, (imageWidth, imageHeight), cv2.INTER_AREA)

#Convert RGB to YUV
def rgbToYuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

#Prepocess image
def preprocessImage(image):
    image = cropImage(image)
    image = resizeImage(image)
    image = rgbToYuv(image)
    return image

#Adjust steering angle and chose image
def chooseImage(dataDir, center, left, right, steeringAngle):
    choice = np.random.choice(3)
    if choice == 0:
        return loadImage(dataDir, left), steeringAngle + 0.3
    elif choice == 1:
        return loadImage(dataDir, right), steeringAngle - 0.3
    return loadImage(dataDir, center), steeringAngle

#Image flip left|right
def randomFlip(image, steeringAngle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steeringAngle = -steeringAngle
    return image, steeringAngle

#Image Translation: Vertical/horizontal
def randomTranslate(image, steeringAngle, rangeX, rangeY):
    transX = rangeX * (np.random.rand() - 0.5)
    transY = rangeY * (np.random.rand() - 0.5)
    steeringAngle += transX * 0.003
    transMatrix = np.float32([[1, 0, transX], [0, 1, transY]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, transMatrix, (width, height))
    return image, steeringAngle

# Generate and add random shadow to the image
def randomShadow(image):
    x1, y1 = imageWidth * np.random.rand(), 0
    x2, y2 = imageWidth * np.random.rand(), imageHeight
    x, y = np.mgrid[0:imageHeight, 0:imageWidth]
    mask = np.zeros_like(image[:, :, 1])
    mask[(y - y1) * (x2 - x1) - (y2 - y1) * (x - x1) > 0] = 1
    cond = mask == np.random.randint(2)
    sRatio = np.random.uniform(low=0.2, high=0.5)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * sRatio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

# Randomly add brightness
def randomBrightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.3 * (np.random.rand() - 0.4)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Image Augmentation
def augment(dataDir, center, left, right, steeringAngle, rangeX=100, rangeY=10):
    image, steeringAngle = chooseImage(dataDir, center, left, right, steeringAngle)
    image, steeringAngle = randomFlip(image, steeringAngle)
    image, steeringAngle = randomTranslate(image, steeringAngle, rangeX, rangeY)
    image = randomShadow(image)
    image = randomBrightness(image)
    return image, steeringAngle

# Generate image batches
def batchGenerator(dataDir, imagePaths, steeringAngles, batchSize, isTraining):
    images = np.empty([batchSize, imageHeight, imageWidth, imageChannels])
    steers = np.empty(batchSize)
    while True:
        i = 0
        for index in np.random.permutation(imagePaths.shape[0]):
            center, left, right = imagePaths[index]
            steeringAngle = steeringAngles[index]
            # Augmentation
            if isTraining and np.random.rand() < 0.5:
                image, steeringAngle = augment(dataDir, center, left, right, steeringAngle)
            else:
                image = loadImage(dataDir, center)
            # Add Image, steering angle for the batch
            images[i] = preprocessImage(image)
            steers[i] = steeringAngle
            i += 1
            if i == batchSize:
                break
        yield images, steers

# Load data from CSV file
def loadData(dataDir,testSize):
    dataDf = pd.read_csv(os.path.join(os.getcwd(), dataDir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = dataDf[['center', 'left', 'right']].values
    y = dataDf['steering'].values
    XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=testSize, random_state=0)
    return XTrain, XValid, yTrain, yValid


#Building Neural Network
def buildModel(keepProb):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5, input_shape=inputShape))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))  
    model.add(Dense(1))
    model.summary()
    return model

#Train Model
def trainModel(model, dataDir, testSize, nbEpoch, samplesPerEpoch, batchSize, saveBestOnly, learningRate):
    XTrain, XValid, yTrain, yValid=loadData(dataDir,testSize)
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='valLoss', verbose=1, saveBestOnly=saveBestOnly, mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learningRate))
    history = model.fit(
        batchGenerator(dataDir, XTrain, yTrain,batchSize, True),
        steps_per_epoch=samplesPerEpoch,
        epochs=nbEpoch,
        validation_data=batchGenerator(dataDir, XValid, yValid, batchSize, False),
        validation_steps=len(XValid) // batchSize,
        callbacks=[checkpoint],
        verbose=1)
    return history

#String to Boolean
def strToBool(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

#Main
def main():
    # Define parameters
    dataDir = 'data'
    testSize = 0.2
    nbEpoch = 10
    samplesPerEpoch = 10000
    batchSize = 32
    saveBestOnly = True
    learningRate = [1.0e-1,1.0e-2,1.0e-3,1.0e-4]
    
    for lr in range(len(learningRate)):
        
        lr=learningRate[lr]
    
        # Print parameters
        print('-' * 30)
        print('Parameters')
        print('-' * 30)
        print(f'dataDir := {dataDir}')
        print(f'testSize := {testSize}')
        print(f'keepProb := {keepProb}')
        print(f'nbEpoch := {nbEpoch}')
        print(f'samplesPerEpoch := {samplesPerEpoch}')
        print(f'batchSize := {batchSize}')
        print(f'saveBestOnly := {saveBestOnly}')
        print(f'learningRate := {learningRate}')
        print('-' * 30)
    
        # Continue with the rest of your code using these parameters
        data = loadData(dataDir, testSize)
        model = buildModel(keepProb)
        trainModel(model, dataDir,testSize ,nbEpoch, samplesPerEpoch, batchSize, saveBestOnly, learningRate)

if __name__ == '__main__':
    main()

