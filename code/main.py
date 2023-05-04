from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math
from preprocess import get_data
from sklearn import preprocessing


class Model(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(Model, self).__init__()

        #Initialize hyperparameters
        self.num_classes = num_classes

        #Initialize the model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="leaky_relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="leaky_relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(self.num_classes, activation="softmax")
        ])


    def call(self, inputs):
        return self.model(inputs)


def run_tests(model, X_test, Y_test):
    X_test = np.reshape(X_test, (-1, *X_test.shape[-2:]))
    Y_test = np.reshape(Y_test, (-1))

    probs = model.predict(X_test, verbose=0)
    diff = np.argmax(probs, axis=1) - Y_test
    accuracy = np.where(diff == 0, 1, 0)

    group_acc = np.array_split(diff, diff.shape[0]//4)
    group_acc = np.apply_along_axis((lambda x: 1 if np.all(x == 0) else 0), axis=1, arr=group_acc)

    return np.mean(accuracy), np.mean(group_acc)

def create_model(input_shape, ohe_size):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="leaky_relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="leaky_relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(ohe_size, activation="softmax")
        ])
    return model

def main():
    #Import and reshape
    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data(.3, "./../processed_data/")

    char_encoder = preprocessing.LabelEncoder().fit(np.concatenate((Y_train.reshape(-1), Y_test.reshape(-1), Y_val.reshape(-1))))
    ohe_size = np.max(char_encoder.transform(np.concatenate((Y_train.reshape(-1), Y_test.reshape(-1), Y_val.reshape(-1)))))
    orig_shapes = [Y_train.shape[0], Y_test.shape[0], Y_val.shape[0]]
    Y_train = char_encoder.transform(Y_train.reshape(-1)).reshape((orig_shapes[0], 4))
    Y_test = char_encoder.transform(Y_test.reshape(-1)).reshape((orig_shapes[1], 4))
    Y_val = char_encoder.transform(Y_val.reshape(-1)).reshape((orig_shapes[2], 4))

    #Convert input to properly shaped and typed tensors
    X_train = tf.convert_to_tensor(np.asarray(np.reshape(X_train, (-1, *X_train.shape[-2:]))).astype(np.float32))
    X_val   = tf.convert_to_tensor(np.asarray(np.reshape(X_val  , (-1, *X_val.shape[-2:]))).astype(np.float32))
    Y_train = tf.convert_to_tensor(np.asarray(np.reshape(Y_train, (-1))))
    Y_val   = tf.convert_to_tensor(np.asarray(np.reshape(Y_val  , (-1))))

    #Don't reshape test, because we need to preserve CAPTCHAs but convert to tensor
    X_test  = tf.convert_to_tensor(np.asarray(X_test).astype(np.float32))
    Y_test  = tf.convert_to_tensor(np.asarray(Y_test))

    #Expand Dims
    X_train = tf.expand_dims(X_train, axis=-1)
    X_val   = tf.expand_dims(X_val  , axis=-1)

    #One hot encode
    Y_val   = tf.one_hot(Y_val  , depth=31, axis=-1)
    Y_train = tf.one_hot(Y_train, depth=31, axis=-1)

    input_shape=(X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])

    # Compile and fit model
    model = create_model(input_shape, ohe_size)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(.0001),
                  metrics= [
                    tf.keras.metrics.CategoricalCrossentropy()
                    ])
    model.fit(
        X_train,
        Y_train,
        epochs=3,
        batch_size=265,
        validation_data=(X_val, Y_val)
    )

    # Save model for future testing
    model.save('./../models/segmented_2', save_format="h5")

    model = tf.keras.models.load_model("./../models/segmented_2")
    print(model.summary())

    # Run random CAPTCHA test
    output = model.predict(X_test[6], verbose=0)
    print("Secret CAPTCHA:")
    print(char_encoder.inverse_transform(Y_test[6]))
    print("Model guess:")
    print(char_encoder.inverse_transform(np.argmax(output, axis=1)))

    # Print individual and captcha-based accuracy
    indiv_acc, group_acc = run_tests(model, X_test, Y_test)
    print(f"Individual accuracy: {round(100*indiv_acc, 2)}% \nCaptcha accuracy: {round(100*group_acc, 2)}%")


if __name__ == '__main__':
    main()