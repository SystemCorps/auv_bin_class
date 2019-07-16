import tensorflow as tf
import keras
import os
from glob import glob
import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications import MobileNetV2
from keras.preprocessing import image
#from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import pickle

base_model = MobileNetV2(weights='imagenet', include_top=False)
#base_model2 = MobileNetV2(weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)
pred = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=pred)



for layer in model.layers:
    layer.trainable = False

    if layer.name == 'dense_1':
        layer.trainable = True
        
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
test_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('D:\Data\net\training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_generator = test_datagen.flow_from_directory('D:\Data\net\test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

valid_generator = valid_datagen.flow_from_directory('D:\Data\net\valid',
                                              target_size = (224, 224),
                                              batch_size = 32,
                                              class_mode = 'binary')

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

step_size_train = train_generator.n//train_generator.batch_size

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch = step_size_train,
                              epochs = 10)


with open('./history/train_history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


model_json = model.to_json()
with open('./model/test.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights("./model/test.h5")