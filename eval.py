from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




batch_size = 32

base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
pred = Dense(2, activation='softmax', use_bias=True)(x)

model = Model(inputs=base_model.input, outputs=pred)
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

pred_list = []

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_generator = test_datagen.flow_from_directory('D:/Data/clearness/valid',
                                            target_size = (224, 224),
                                            batch_size = batch_size,
                                            shuffle=False,
                                            class_mode = 'categorical')

labels = test_generator.classes

for i in range(11,21):
    model_path = './results/check/check_%d' % i
    model.load_weights(model_path)

    


    test_steps = test_generator.n//test_generator.batch_size
    test = model.evaluate_generator(generator=test_generator,
                                    steps=test_steps)

    preds = model.predict_generator(generator=test_generator,
                                    steps=test_steps)

    pred_list.append(preds)

    print(test)

t1 = np.argmax(pred_list[0])


