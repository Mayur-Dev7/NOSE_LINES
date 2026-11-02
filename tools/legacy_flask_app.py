from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import json
#import jsonpickle
#import cv2
# Keras


import tensorflow as tf
from tensorflow import keras

import tensorflow.keras as keras

from keras.applications.imagenet_utils import preprocess_input
#from keras.models import load_model
from keras.models import save_model, load_model

#from tensorflow.keras.models import load_model, model
from keras.preprocessing import image







# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Response
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
from flask import jsonify

# Define a flask app
import logging
import logging.handlers as handlers
import os


#akash code 



#%cd /content/drive/My Drive/ML/prasadmuzzledetector/Mask_RCNN
'''tensorflow_version 1.x #1.15.2
import keras
#!pip install keras==2.1.6 2.1.5
if keras.__version__ != '2.1.0':
  !pip uninstall keras
  #!pip uninstall tensorflow
  #pip install tensorflow==1.15.2
  !pip install keras==2.1.0
else:
  print(keras.__version__ )'''
#from google.colab import drive
#drive.mount('/content/drive')
import os
os.chdir('../Mask_RCNN')
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
#matplotlib inline
from os import listdir
from xml.etree import ElementTree
class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background)
    #  hump + BG
    NUM_CLASSES = 1+1
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 150
    
    # Learning rate
    LEARNING_RATE=0.001
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10
config = myMaskRCNNConfig()


#Loading the model in the inference mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
model_path = '/home/sujit/Desktop/gangadhar/crop11dec.h5'

# loading the trained weights o the custom dataset
model.load_weights(model_path, by_name=True)


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from PIL import Image
import cv2
import numpy as np
#akash

def loggerfun():
    myhost = os.uname()[1]
    logger = logging.getLogger('flask')
    logger.setLevel(logging.INFO)
    d={'hostname':myhost,'class':'flask'}
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(hostname)s- %(threadName)s- %(levelname)s -%(class)s- %(message)s',datefmt="%y-%b-%d %H:%M:%S")
    logHandler = handlers.TimedRotatingFileHandler('flask.log', when='D', interval=1, backupCount=100)
    logHandler.setFormatter(formatter)
    if (logger.hasHandlers()):
         logger.handlers.clear()
    logger.addHandler(logHandler)
    return logger,d


app = Flask(__name__)


# Load your trained model

# load the saved model back
model = load_model('/home/sujit/Desktop/gangadhar/cowrecupdatedw.h5', compile=False)


#model._make_predict_function()  




with open('/home/sujit/Desktop/gangadhar/classes.json') as json_file: 
    classesdict = json.load(json_file) 
classesdict = {int(k):v for k,v in classesdict.items()}
# model.predict()      
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    logger,d = loggerfun()
    try:
        test_image = image.load_img(img_path, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
       # image="/content/drive/My Drive/ML/prasadmuzzledetector/muzzledataset/test/images/00001.jpg" #take here post image
        test_image = load_img(test_image) 
        img = img_to_array(test_image)
        # detecting objects in the image
        result= model.detect([img])
        # print(r)
        r = result[0]
        if r['rois'].size > 0:
          ax = pyplot.gca()
          y1, x1, y2, x2 = r['rois'][0]
          #     # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          test_image=test_image.crop((x1,y1,x2,y2)) #croping img

        #canny img
          #img2 = cv2.imread(image,0) #img name,0=gray
          test_image = np.array(test_image)
          sobalx=cv2.Sobel(test_image,cv2.CV_64F,1,0) #img,datatype,x direction,y deirection
          sobalx=np.uint8(np.absolute(sobalx))
          kernel = np.ones((5,5),np.float32)/25
          dst = cv2.filter2D(sobalx,-1,kernel)
          IMG_SIZE =200
          new_array = cv2.resize(dst, (IMG_SIZE, IMG_SIZE))
          edges = cv2.Canny(new_array,100,200) #send this image to detect









        test_image = np.expand_dims(test_image, axis = 0)
        result1 = model.predict_proba(test_image/255)
        result = model.predict(test_image)
        logger.info("Succesfully predicted image", extra=d, exc_info=True)
        predicted_classes = np.argmax(result1,axis=1)
        confidenceValue = result1[0][predicted_classes[0]]*100
        c='confidence : {:.2f}%'.format(result1[0][predicted_classes[0]]*100)
        if confidenceValue >= 10:
                max_weight = np.argmax(result[0])
                return "cow"+"-"+ str(classesdict[int(max_weight)])+" "+c
                return jsonify("cow"+"-"+ str(classesdict[int(max_weight)])+" "+c)
        else:
            return "Cant Predict the given Image"
    except:
        
        logger.error("Failed to predict image",extra = d,exc_info=True)

    


@app.route('/', methods=['GET']) #default home page / or /index
def index():
    # Main page
    try:
        return render_template('index.html')
    except:
        logger,d = loggerfun()
        logger.error("Failed to load to index",extra = d,exc_info=True)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    logger,d = loggerfun()
    try:
        if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds = model_predict(file_path, model)
            logger.info("succesfully uploaded image",extra=d,exc_info=True)
            return preds
        return "hello"
    except:
       
        logger.error("Failed to upload image",extra = d,exc_info=True)


                   

if __name__ == '__main__':
    app.run(debug=True)

