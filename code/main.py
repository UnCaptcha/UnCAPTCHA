from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math
from preprocess import get_data

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        #Initialize hyperparameters
        self.num_classes = 36
        self.learning_rate = 0.001
        self.image_size = (24,24,1)

        #Initialize the model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), padding="same"),
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


    def loss(self, logits, labels):
        return tf.keras.losses.SparseCategoricalCrossentropy(logits, labels)


    def accuracy(self, logits, labels):
        pass


def train(model, train_inputs, train_labels):
    pass


def test(model, test_inputs, test_labels):
    pass


def visualize_loss(losses):
    """
    Taken from HW3 support code:
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Taken from HW3 Support Code:
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    #Import and reshape
    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data(.3, "./../processed_data/")
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

    model = Model()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics= [
                      tf.keras.metrics.BinaryCrossentropy()
                  ])
    model.fit(
        X_train,
        Y_train,
        epochs=3,
        batch_size=500,
        validation_data=(X_val, Y_val)
    )


if __name__ == '__main__':
    main()