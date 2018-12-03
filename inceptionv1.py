#!/usr/bin/python

# General
import os
import cv2
import math
import numpy as np

# sklearn
from sklearn.model_selection import train_test_split

# Keras
import keras
from keras.layers.core import Layer
import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dropout, Dense, Input
from keras.layers import concatenate, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, Flatten
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils

# Tensorflow
import tensorflow as tf

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

"""
@params
X: input from previous layer
filter_1x1_left: output_size for the left side CNN-1x1
filter_1x1_3x3: output_size for reduction CNN-1x1 -> CNN-3x3
filter_1x1_5x5: output_size for reduction CNN-1x1 -> CNN-5x5
filter_3x3: output_size for CNN-3x3 (layer 2)
filter_5x5: output_size for CNN-5x5 (layer 2)
filter_1x1_max_pool: output_size for CNN-1x1 after MaxPool (layer 2)
"""
def inception_module(X, filter_1x1_left, filter_1x1_3x3, filter_1x1_5x5, filter_3x3, filter_5x5, filter_1x1_max_pool, name=None):
    # Inception Module abstraction

    # First Layer: X = input from previous layer
    # CNN 1x1 on the left, connects directly to the concatenation layer
    cnn_1x1 = Conv2D(filter_1x1_left, (1, 1), padding="same", activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    # CNN 1x1 before the CNN 3x3 of the 2nd layer
    cnn_1x1_3x3 = Conv2D(filter_1x1_3x3, (1, 1), padding="same", activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    # CNN 1x1 before the CNN 5x5 of the 2nd layer
    cnn_1x1_5x5 = Conv2D(filter_1x1_5x5, (1, 1), padding="same", activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    # Max Pooling of 3x3 before the CNN 1x1 of the 2nd layer
    max_pool = MaxPool2D((3, 3), strides=(1, 1), padding="same")(X)

    # Second Layer: Inputs are from the first Layer (1x1 or Max Pool)
    # CNN 3x3 on the left, input from CNN 1x1
    cnn_3x3 = Conv2D(filter_3x3, (3, 3), padding="same", activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(cnn_1x1_3x3)
    # CNN 5x5 in the middle, input from CNN 1x1
    cnn_5x5 = Conv2D(filter_5x5, (5, 5), padding="same", activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(cnn_1x1_5x5)
    cnn_1x1_pooled = Conv2D(filter_1x1_max_pool, (1, 1), padding="same", activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(max_pool)

    # Concatenate the layers into the output of the Inception module
    filter_concatenation = concatenate([cnn_1x1, cnn_3x3, cnn_5x5, cnn_1x1_pooled], axis=3, name=name)
    return filter_concatenation

# From the Table #1 of "Going deeper with inception"
# Architecture: http://joelouismarino.github.io/images/blog_images/blog_googlenet_keras/googlenet_diagram.png
def create_model(num_samples=11):
    # Layer0: Input layer (as defined in the paper, RGB 224x224)
    input_layer = Input(shape=(224, 224, 3))

    # Layer1: Conv 7x7/2 -> 112x112x64;
    layer_1 = Conv2D(64, (7, 7), padding="same", strides=(2, 2), activation="relu", name="layer_1_conv", kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)

    # Layer2: Maxpool 3x3/2 -> 56x56x64
    layer_2 = MaxPool2D((3, 3), padding="same", strides=(2, 2), name="layer_2_max_pool")(layer_1)

    # Layer3: Conv 3x3/1 -> 56x56x192
    layer_3 = Conv2D(64, (1, 1), padding="same", strides=(1, 1), activation="relu", name="layer_3_conv")(layer_2)

    # Layer4: Conv 3x3/1 -> 56x56x192
    layer_4 = Conv2D(192, (3, 3), padding="same", strides=(1, 1), activation="relu", name="layer_4_conv")(layer_3)

    # Layer5: Maxpool 3x3/2 -> 28x28x192
    layer_5 = MaxPool2D((3, 3), padding="same", strides=(2, 2), name="layer_5_max_pool")(layer_4)

    # Layer6: Inception 3a -> 28x28x256, depth = 2
    layer_6 = inception_module(layer_5, 64, 96, 16, 128, 32, 32, name="layer_6_inception_3a")

    # Layer7: Inception 3b -> 28x28x480, depth = 2
    layer_7 = inception_module(layer_6, 128, 128, 32, 192, 96, 64, name="layer_7_inception_3b")

    # Layer8: Maxpool 3x3/2 -> 14x14x480
    layer_8 = MaxPool2D((3, 3), padding="same", strides=(2, 2), name="layer_8_maxpool")(layer_7)

    # Layer9: Inception 4a -> 14x14x514
    layer_9 = inception_module(layer_8, 192, 96, 16, 208, 48, 64, name="layer_9_inception_4a")
    
    # Parallel Inception + Auxiliary Classifier
    # Layer10: Inception 4b -> 14x14x514
    layer_10 = inception_module(layer_9, 160, 112, 24, 224, 64, 64, name="layer_10_inception_4b")

    # Auxiliary Classifier:
    layer_10_1 = AveragePooling2D((5, 5), strides=3)(layer_9)
    layer_10_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(layer_10_1)
    layer_10_3 = Flatten()(layer_10_2)
    layer_10_4 = Dense(1024, activation='relu')(layer_10_3)
    layer_10_5 = Dropout(0.7)(layer_10_4)
    layer_10_6 = Dense(num_samples, activation='softmax', name='auxilliary_output_1')(layer_10_5)

    # Layer11: Inception 4c -> 14x14x514
    layer_11 = inception_module(layer_10, 128, 128, 24, 256, 64, 64, name="layer_11_inception_4c")

    # Layer12: Inception 4d -> 14x14x528
    layer_12 = inception_module(layer_11, 112, 144, 32, 288, 64, 64, name="layer_12_inception_4d")

    # Parallel Inception + Auxiliary Classifier
    # Layer13: Inception 4e -> 14x14x832
    layer_13 = inception_module(layer_12, 256, 160, 32, 320, 128, 128, name="layer_13_inception_4e")

    # Auxiliary Classifier
    layer_13_1 = AveragePooling2D((5, 5), strides=3)(layer_12)
    layer_13_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(layer_13_1)
    layer_13_3 = Flatten()(layer_13_2)
    layer_13_4 = Dense(1024, activation='relu')(layer_13_3)
    layer_13_5 = Dropout(0.7)(layer_13_4)
    layer_13_6 = Dense(num_samples, activation='softmax', name='auxilliary_output_2')(layer_13_5)

    # Layer14: Maxpool 3x3/2 -> 7x7x832
    layer_14 = MaxPool2D((3, 3), padding='same', strides=(2, 2), name="layer_14_maxpool")(layer_13)

    # Layer15: Inception 5a -> 7x7x832
    layer_15 = inception_module(layer_14, 256, 160, 32, 320, 128, 128, name="layer_15_inception_5a")

    # Layer16: Inception 5b -> 7x7x1024
    layer_16 = inception_module(layer_15, 384, 192, 48, 384, 128, 128, name="layer_16_inception_5b")

    # Layer17: Avg. Pool -> 1x1x1024
    layer_17 = GlobalAveragePooling2D(name="layer_17_global_avg_pool")(layer_16)

    # Layer18: Dropout Pool (40%) -> 1x1x1024
    layer_18 = Dropout(0.4)(layer_17)

    # Layer19: Softmax -> 1x1x1000
    layer_19 = Dense(num_samples, activation='softmax', name="layer_19_dense")(layer_18)

    # Layer20: Output
    model = Model(input_layer, [layer_19, layer_10_6, layer_13_6], name='inception_v1')
    return model

def load_and_process_data(path, sample_size_max=None, image_size=128):
    dataset = []
    labels = []
    labels_name = []
    print ("Importing MIO-TCD images dataset...")
    # Loop through the dataset directory with all the labels as subdirectories
    for folder_name in os.listdir(path):
        # Discard hidden files and folders of linux/MacOS
        if (not folder_name.startswith(".")):
            labels_name.append(folder_name)
            sample_counter = 0
            for file_name in os.listdir(path + "/" + folder_name):
                if (not file_name.startswith(".")):
                    # Import image
                    img = cv2.imread(path + "/" + folder_name + "/" + file_name)
                    # Process, transform into features, add to global dataset
                    # Resize every image to a fixed, save raw input
                    img = cv2.resize(img, (image_size, image_size))
                    dataset.append(img)
                    labels.append(folder_name)
                    sample_counter = sample_counter + 1
                if (sample_size_max is not None):
                    if (sample_counter == sample_size_max):
                        break
    dataset = np.array(dataset)
    labels = np.array(labels)
    print ("# of Samples: " + str(dataset.shape[0]))
    print ("Importating done!\n")
    # Split into train/test
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.20)

    Y_train_uniques, Y_train_int_labels = np.unique(Y_train, return_inverse=True)
    Y_test_uniques, Y_test_int_labels = np.unique(Y_test, return_inverse=True)

    Y_train = np_utils.to_categorical(Y_train_int_labels, 11)
    Y_test = np_utils.to_categorical(Y_test_int_labels, 11)
    
    X_train = X_train.astype('float32')
    Y_test = Y_test.astype('float32')

    # preprocess data
    X_train = X_train / 255.0
    Y_test = Y_test / 255.0

    return X_train, X_test, Y_train, Y_test

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

if __name__ == "__main__":
    print ("###")
    print ("Classification Task with Deep Learning")
    print ("###\n")
    
    # Import dataset, resize images, return train/test datasets
    X_train, X_test, Y_train, Y_test = load_and_process_data("./data/classification", sample_size_max=200, image_size=224)
    
    # Create our neural network model (GoogLeNet, Inception v1)
    print ("Creating our model... ")
    model = create_model()
    model.summary()
    print ("Model is ready!\n")

    print ("Learning settings: ")
    epochs = 25
    initial_lrate = 0.01
    sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)
    lr_sc = LearningRateScheduler(decay, verbose=1)
    print ("# of epoch: " + str(epochs))
    print ("learning_rate: " + str(initial_lrate) + "\n")

    print ("Fitting our model...")
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])
    history = model.fit(X_train, [Y_train, Y_train, Y_train], validation_data=(X_test, [Y_test, Y_test, Y_test]), epochs=epochs, batch_size=256, callbacks=[lr_sc])
    model.save("1500.h5")
