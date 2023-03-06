#!/usr/bin/env python
import os

### For macOS AMD graphics
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["RUNFILES_DIR"] = "/Users/kaansenparlak/Library/Python/3.8/share/plaidml"
os.environ["PLAIDML_NATIVE_PATH"] = "/Users/kaansenparlak/Library/Python/3.8/lib/libplaidml.dylib"

import keras
import numpy as np
from keras import layers
from configs import *

x_train = np.load("np/x_train.npy")
y_train = np.load("np/y_train.npy")
x_val = np.load("np/x_val.npy")
y_val = np.load("np/y_val.npy")


def get_cnn(width=width, height=height, depth=depth, channel=channel):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, channel))

    x = layers.Conv3D(filters=8, kernel_size=(7, 7, 1), strides=2, activation=None)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=16, kernel_size=(5, 5, 1), strides=2, activation=None)(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=2, activation=None)(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = x

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


def get_model(width, height, depth, channel):

    cnn_model = get_cnn(width, height, depth, channel)

    inputs = cnn_model.input
    outputs = cnn_model.output

    model = keras.Model(inputs, outputs, name="mixed")
    return model


# Build model.
model = get_model(width, height, depth, channel)
model.summary()

loss = "binary_crossentropy" if classification else "mean_squared_error"

# Compile model.
model.compile(
    loss=loss,
    optimizer=keras.optimizers.Adamax(),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

x_train = keras.backend.constant(x_train); y_train = keras.backend.constant(y_train)
x_val = keras.backend.constant(x_val); y_val = keras.backend.constant(y_val)
x_train = x_train.eval(); y_train = y_train.eval(); x_val = x_val.eval(); y_val = y_val.eval()

val_to_fit = x_val
train_to_fit = x_train

from augmented import generator
from unet.build_model import build_model

# Create image data generators for both train and validation sets
image_aug = generator.customImageDataGenerator(
    rotation_range=90,
)

X_train_datagen = image_aug.flow(train_to_fit, y_train, batch_size=batch_size, seed=depth) # set equal seed number
train_generator = zip(X_train_datagen, y_train)

# Train the model, doing validation at the end of each epoch
model.fit_generator(
    X_train_datagen,
    # (train_to_fit, y_train),
    validation_data=(val_to_fit, y_val),
    validation_steps=len(y_val),
    steps_per_epoch=len(y_train),
    epochs=epochs,
    shuffle=True,
    verbose=2,
    # batch_size=batch_size,
    callbacks=[checkpoint_cb],
)
