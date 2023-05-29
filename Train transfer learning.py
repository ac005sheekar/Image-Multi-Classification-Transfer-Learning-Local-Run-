###           Written and Executed By:            ####
###               Sheekar Banerjee                ####
###             AI Engineering Lead               ####


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import sklearn.metrics as metrics
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16


num_classes=4  #number of classes
IMAGE_SHAPE = [224, 224]  #Shape of the images
batch_size=32   #Batch Size
epochs = 100    #Number of epochs for iteration


#
vgg = VGG16(input_shape = (224,224,3), weights = "imagenet", include_top = False)
for layer in vgg.layers:
    layer.trainable = False


x = Flatten()(vgg.output)
x = Dense(128, activation = "relu")(x)
x = Dense(64, activation = "relu")(x)
x = Dense(num_classes, activation = "softmax")(x)
model = Model(inputs = vgg.input, outputs = x)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



trdata = ImageDataGenerator()
train_data_gen = trdata.flow_from_directory(directory="C:/Users/SHEEKAR/PycharmProjects/ML Dense/Alzheimers Dataset/train",
                                            target_size=(224,224),
                                            shuffle=False,
                                            class_mode="categorical")
tsdata = ImageDataGenerator()
test_data_gen = tsdata.flow_from_directory(directory="C:/Users/SHEEKAR/PycharmProjects/ML Dense/Alzheimers Dataset/test",
                                           target_size=(224,224),
                                           shuffle=False,
                                           class_mode="categorical")


training_steps_per_epoch = np.ceil(train_data_gen.samples / batch_size)
validation_steps_per_epoch = np.ceil(test_data_gen.samples / batch_size)
#Model.fit
model.fit_generator(train_data_gen,
                    steps_per_epoch = training_steps_per_epoch,
                    validation_data=test_data_gen,
                    validation_steps=validation_steps_per_epoch,
                    epochs=epochs, verbose=1)
print("Training Completed!")


Y_pred = model.predict(test_data_gen, test_data_gen.samples / batch_size)
val_preds = np.argmax(Y_pred, axis=1)



import sklearn.metrics as metrics
val_trues =test_data_gen.classes

from sklearn.metrics import classification_report
print(classification_report(val_trues, val_preds))



Y_pred = model.predict(test_data_gen, test_data_gen.samples / batch_size)
val_preds = np.argmax(Y_pred, axis=1)
val_trues =test_data_gen.classes
cm = metrics.confusion_matrix(val_trues, val_preds)
print(cm)


keras_file="trans-alzh-Model.h5"
tf.keras.models.save_model(model,keras_file)



