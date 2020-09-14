from __future__ import division , print_function
import sys
import os 
import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.models import load_model
from keras import backend
from tensorflow.keras import backend


import tensorflow as tf
global graph
graph = tf.get_default_graph()
from skimage.transform import resize
from flask import Flask,request,render_template,redirect,url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("pneumonia.h5")

@app.route('/',methods = ["GET"] )
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)#gives the current path of this file
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
                
            
            print("prediction",preds)
            
        index = ['NORMAL' , 'PNEUMONIA']
        
        text = "The Prediction is....: " + str(index[preds[0][0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)
        
        
        
    
    
    
        
        
        
    
    
    