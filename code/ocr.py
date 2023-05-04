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


def get_accuracy(model, X_test, Y_test):
    probs = model.predict(X_test, verbose=0)
    output = np.argmax(probs, axis=2)

    prediction = np.asarray([np.asarray(tf.unique_with_counts(x)[0]) for x in output])
    Y_test = np.asarray(Y_test)
    count = 0
    total = Y_test.shape[0]
    for x, y in zip(prediction, Y_test):
        if len(x) == len(y) and np.all(np.equal(x, y)): count += 1
    accuracy = count / total

    return accuracy

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
            tf.keras.layers.MaxPool2D(pool_size=(2,3)), # output size: 3,3,128

            tf.keras.layers.Reshape((input_shape[-2] // 12, input_shape[-3] // 8 * 128)),

            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_shape[-3] // 8 * 128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_shape[-3] // 8 * 128, return_sequences=True)),

            tf.keras.layers.Dense(ohe_size+1, activation="softmax")
        ])
    return model

def main():
    #Import and reshape
    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data_ocr(.3, "./processed_whole_data/")

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
    X_test  = tf.expand_dims(X_test , axis=-1)

    input_shape=(X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    print(input_shape)


    def ctc(y_true, y_pred):
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
    model = create_model(input_shape, (ohe_size+1))
    model.compile(loss=ctc,
                    optimizer=tf.keras.optimizers.Adam(.0001))

    model.summary()

    model.fit(
        X_train,
        Y_train,
        epochs=80,
        batch_size=256,
        validation_data=(X_val, Y_val)
    )

    # Save model for future testing
    model.save('./models/ocr', save_format="h5")

    loaded_model = tf.keras.models.load_model("./models/ocr", custom_objects={'ctc': ctc})
    print(loaded_model.summary())

    # Run random CAPTCHA test
    output = model.predict(X_test[1:2], verbose=0)
    print("Secret CAPTCHA:")
    print(char_encoder.inverse_transform(Y_test[1]))
    print("Model guess:")
    output = np.transpose(output, axes=[1, 0, 2]).squeeze()
    print(output)
    print(np.argmax(output, axis=1))

    prediction = tf.unique_with_counts(np.argmax(output, axis=1))
    print(prediction)

    print("Accuracy:")
    print(get_accuracy(model, X_test, Y_test))


if __name__ == '__main__':
    main()