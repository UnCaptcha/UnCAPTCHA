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


    # def loss(self, logits, labels):
    #     return tf.keras.losses.SparseCategoricalCrossentropy(logits, labels)


    # def accuracy(self, logits, labels):
    #     pass


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
    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data(.3, "./../barcoded_data/")

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
    
    # # Compile and fit model
    # model = create_model(input_shape, ohe_size)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=tf.keras.optimizers.Adam(.0001),
    #               metrics= [
    #                 tf.keras.metrics.CategoricalCrossentropy()
    #                 ])
    # model.fit(
    #     X_train,
    #     Y_train,
    #     epochs=5,
    #     batch_size=256,
    #     validation_data=(X_val, Y_val)
    # )

    # # Save model for future testing
    # model.save('./../models/barcoded')

    model = tf.keras.models.load_model("./../models/barcoded")
    print(model.summary())

    # Run random CAPTCHA test
    output = model.predict(X_test[7], verbose=0)
    print("Secret CAPTCHA:")
    print(char_encoder.inverse_transform(Y_test[6]))
    print("Model guess:")
    print(char_encoder.inverse_transform(np.argmax(output, axis=1)))

    # Print individual and captcha-based accuracy
    indiv_acc, group_acc = run_tests(model, X_test, Y_test)
    print(f"Individual accuracy: {round(100*indiv_acc, 2)}% \nCaptcha accuracy: {round(100*group_acc, 2)}%")


if __name__ == '__main__':
    main()