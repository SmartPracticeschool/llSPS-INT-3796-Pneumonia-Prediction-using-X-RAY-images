# -*- coding: utf-8 -*-

#Creating and training model for predicting output
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

project=Sequential()
project.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
project.add(MaxPooling2D(pool_size=(2,2)))
project.add(Flatten())
project.add(Dense(output_dim =128,init ='uniform' , activation='relu'))
project.add(Dense(output_dim=1, init='uniform',activation='sigmoid'))


from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2, zoom_range=0.2 ,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory(r"C:\Users\lavke\Downloads\Dataset_Chest_XRay\train",target_size=(64,64), batch_size=32 ,class_mode='binary')
x_test=test_datagen.flow_from_directory(r"C:\Users\lavke\Downloads\Dataset_Chest_XRay\test" ,target_size=(64,64),batch_size=32 , class_mode='binary')


print(x_train.class_indices)

project.compile(loss= 'binary_crossentropy', optimizer="adam" , metrics=["accuracy"])
project.fit_generator(x_train , steps_per_epoch =500,epochs =10,validation_data=x_test ,validation_steps= 100)
project.save("pneumonia.h5")


#Predicting output from our trained model
from keras.preprocessing import image
img=image.load_img(r"C:\Users\lavke\Downloads\chestimg.jpg" ,target_size=(64,64))


x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)


pred =project.predict_classes(x)
pred