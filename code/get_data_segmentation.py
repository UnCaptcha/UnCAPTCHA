import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import cv2
import os


def retrieve_data(data_path):
    """
    Function to handle reshaping and formatting of data before returning to model.

    Inputs:
    data_path - Path to data to be used

    Outputs:
    X_train, X_test, X_val, Y_train, Y_test, Y_val - Divided data (X) and labels (Y)
    char_encoder - Encoder used to ohe our data
    encoder_size - Number of ohe values used.
    """
    # Import and reshape
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_split_data(data_path)

    # Generate a one hot encoder for the dataset
    char_encoder = preprocessing.LabelEncoder().fit(np.concatenate((Y_train.reshape(-1), Y_test.reshape(-1), Y_val.reshape(-1))))
    encoder_size = np.max(char_encoder.transform(np.concatenate((Y_train.reshape(-1), Y_test.reshape(-1), Y_val.reshape(-1)))))
    orig_shapes  = [Y_train.shape[0], Y_test.shape[0], Y_val.shape[0]]

    # Encode from characters to numbers and reshape all of the labels
    Y_train = char_encoder.transform(Y_train.reshape(-1)).reshape((orig_shapes[0], 4))
    Y_test  = char_encoder.transform(Y_test.reshape(-1)).reshape((orig_shapes[1], 4))
    Y_val   = char_encoder.transform(Y_val.reshape(-1)).reshape((orig_shapes[2], 4))

    # Convert input to properly shaped and typed tensors
    X_train = tf.convert_to_tensor(np.asarray(np.reshape(X_train, (-1, *X_train.shape[-2:]))).astype(np.float32))
    X_val   = tf.convert_to_tensor(np.asarray(np.reshape(X_val  , (-1, *X_val.shape[-2:]))).astype(np.float32))
    Y_train = tf.convert_to_tensor(np.asarray(np.reshape(Y_train, (-1))))
    Y_val   = tf.convert_to_tensor(np.asarray(np.reshape(Y_val  , (-1))))

    # Don't reshape test, because we need to preserve CAPTCHAs but convert to tensor
    X_test  = tf.convert_to_tensor(np.asarray(X_test).astype(np.float32))
    Y_test  = tf.convert_to_tensor(np.asarray(Y_test))

    # Expand Dims
    X_train = tf.expand_dims(X_train, axis=-1)
    X_val   = tf.expand_dims(X_val  , axis=-1)

    # One hot encode
    Y_val   = tf.one_hot(Y_val  , depth=encoder_size, axis=-1)
    Y_train = tf.one_hot(Y_train, depth=encoder_size, axis=-1)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def get_split_data(processed_data_path):
    """
    Retrieves split data from their respective folders after preprocessing has been done

    processed_data_path - Path of preprocessed data
    """
    captcha = []
    captcha_labels = []
    count = 0

    arrs = ([], [], [], [], [], [])
    for i, state in enumerate(["train", "test", "val"]):

        for image_path in sorted(os.listdir(processed_data_path + state + "/")):
            count += 1
            image = cv2.imread(processed_data_path + state + "/" + image_path, cv2.IMREAD_GRAYSCALE)

            label = image_path.split('.')[0][-1]
            captcha.append(image)
            captcha_labels.append(label)

            if count % 4 == 0:
                arrs[i*2].append(captcha)
                arrs[i*2 + 1].append(captcha_labels)
                captcha = []
                captcha_labels = []

    return np.asarray(arrs[0]), np.asarray(arrs[1]), np.asarray(arrs[2]), np.asarray(arrs[3]), np.asarray(arrs[4]), np.asarray(arrs[5])
