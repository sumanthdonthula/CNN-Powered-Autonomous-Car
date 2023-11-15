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



# Initialize our server
sio = socketio.Server()
app = Flask(__name__)
model = None
prevImageArray = None
maxSpeed = 30
minSpeed = 10
speedLimit = maxSpeed

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
                speedLimit = minSpeed
            else:
                speedLimit = maxSpeed
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
        help='Image Folder Path'
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
