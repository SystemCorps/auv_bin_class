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
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard, ReduceLROnPlateau

import pickle


class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def main():
    batch_size = 32

    base_model = MobileNetV2(weights='imagenet', include_top=False)
    base_model2 = MobileNetV2(weights='imagenet')


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    pred = Dense(2, activation='softmax', use_bias=True)(x)

    model = Model(inputs=base_model.input, outputs=pred)

    #model = base_model


    for layer in model.layers:
        layer.trainable = False

        if layer.name == 'dense_1':
            layer.trainable = True
        
    train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory('D:/Data/clearness/train',
                                                     target_size = (224, 224),
                                                     batch_size = batch_size,
                                                     shuffle=True,
                                                     class_mode = 'categorical')

    test_generator = test_datagen.flow_from_directory('D:/Data/clearness/test',
                                                target_size = (224, 224),
                                                batch_size = batch_size,
                                                shuffle=False,
                                                class_mode = 'categorical')

    valid_generator = valid_datagen.flow_from_directory('D:/Data/clearness/valid',
                                                  target_size = (224, 224),
                                                  batch_size = batch_size,
                                                  shuffle = True,
                                                  class_mode = 'categorical')

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    step_size_train = train_generator.n//train_generator.batch_size

    #history = model.fit_generator(generator=train_generator,
    #                              steps_per_epoch = step_size_train,
    #                              epochs = 10)



    loss = LossHistory()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0, mode='auto')
    num_trial = 5
    tensorboard = TensorBoard(log_dir='./results/logs_{}'.format(num_trial),
                              histogram_freq=0, batch_size=batch_size, 
                              write_graph=True, write_grads=False, write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None,
                              update_freq='epoch')


    filepath = './results/check/check_{}'.format(num_trial)
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


    num_epoch = 1
    val_period = 10


    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch = step_size_train,
                                  epochs = num_epoch,
                                  validation_data = valid_generator,
                                  validation_steps = val_period,
                                  callbacks=[checkpointer, loss, reduce_lr, tensorboard])

    # train_generator.n//train_generator.batch_size
    test_steps = test_generator.n//test_generator.batch_size
    history_ev = model.evaluate_generator(generator=test_generator,
                                          steps=test_steps)

    print(history_ev)

    print("Done")

    preds = model.predict_generator(generator=test_generator,
                                    steps=test_steps)



if __name__ == '__main__':
    main()