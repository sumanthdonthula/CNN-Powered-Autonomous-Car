import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
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

# Initialize our server
sio = socketio.Server()
app = Flask(__name__)
model = None
prevImageArray = None
MAX_SPEED = 25
MIN_SPEED = 10
speedLimit = MAX_SPEED

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steeringAngle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)
            image = preprocess(image)
            image = np.array([image])
            steeringAngle = float(model.predict(image, batch_size=1))
            global speedLimit
            if speed > speedLimit:
                speedLimit = MIN_SPEED
            else:
                speedLimit = MAX_SPEED
            throttle = 0.4- steeringAngle**2 - (speed/speedLimit)**2
            sendControl(steeringAngle, throttle)
        except Exception as e:
            print(e)

        if args.imageFolder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            imageFilename = os.path.join(args.imageFolder, timestamp)
            image.save('{}.jpg'.format(imageFilename))
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    sendControl(0, 0)

def sendControl(steeringAngle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steeringAngle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be in the same path.'
    )
    parser.add_argument(
        'imageFolder',
        type=str,
        nargs='?',
        default='',
        help='Image path.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.imageFolder != '':
        print("Creating image folder at {}".format(args.imageFolder))
        if not os.path.exists(args.imageFolder):
            os.makedirs(args.imageFolder)
        else:
            shutil.rmtree(args.imageFolder)
            os.makedirs(args.imageFolder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
