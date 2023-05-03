from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math
from preprocess import get_data_ocr
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

            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50 , return_sequences=True))

            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(256, activation='leaky_relu'),
            # tf.keras.layers.Dense(self.num_classes, activation="softmax")
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
            tf.keras.layers.MaxPool2D(pool_size=(2,1)), # output size: 3,3,128

            tf.keras.layers.Reshape((input_shape[-2] // 4, input_shape[-3] // 8 * 128)),

            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_shape[-3] // 8 * 128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_shape[-3] // 8 * 128, return_sequences=True)),

            tf.keras.layers.Dense(ohe_size+1, activation="softmax")
        ])
    return model

def main():
    #Import and reshape
    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data_ocr(.3, "./../data/")

    char_encoder = preprocessing.LabelEncoder().fit(np.concatenate((Y_train.reshape(-1), Y_test.reshape(-1), Y_val.reshape(-1))))
    ohe_size = np.max(char_encoder.transform(np.concatenate((Y_train.reshape(-1), Y_test.reshape(-1), Y_val.reshape(-1)))))
    orig_shapes = [Y_train.shape[0], Y_test.shape[0], Y_val.shape[0]]
    Y_train = char_encoder.transform(Y_train.reshape(-1)).reshape((orig_shapes[0], 4))
    Y_test = char_encoder.transform(Y_test.reshape(-1)).reshape((orig_shapes[1], 4))
    Y_val = char_encoder.transform(Y_val.reshape(-1)).reshape((orig_shapes[2], 4))

    #Convert input to properly shaped and typed tensors
    X_train  = tf.convert_to_tensor(np.asarray(X_train).astype(np.float32))
    Y_train  = tf.convert_to_tensor(np.asarray(Y_train))
    X_val  = tf.convert_to_tensor(np.asarray(X_val).astype(np.float32))
    Y_val  = tf.convert_to_tensor(np.asarray(Y_val))

    #Don't reshape test, because we need to preserve CAPTCHAs but convert to tensor
    X_test  = tf.convert_to_tensor(np.asarray(X_test).astype(np.float32))
    Y_test  = tf.convert_to_tensor(np.asarray(Y_test))

    #Expand Dims
    X_train = tf.expand_dims(X_train, axis=-1)
    X_val   = tf.expand_dims(X_val  , axis=-1)

    #One hot encode
    #Y_val   = tf.one_hot(Y_val  , depth=ohe_size, axis=-1)
    #Y_train = tf.one_hot(Y_train, depth=ohe_size, axis=-1)

    input_shape=(X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    print(input_shape)


    def ctc(y_true, y_pred):
        print(y_true.shape)
        y_true = tf.cast(y_true, tf.int32)
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        #input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        #label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=False, dtype=tf.int32)

        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, label_length=label_length, input_length=input_length)
    # Compile and fit model
    model = create_model(input_shape, ohe_size)
    model.compile(loss=ctc,
                  optimizer=tf.keras.optimizers.Adam(.0001))

    print(Y_train.shape)
    model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=256,
        validation_data=(X_val, Y_val)
    )

    # Save model for future testing
    model.save('./../models/barcoded')

    model = tf.keras.models.load_model("./../models/barcoded")
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