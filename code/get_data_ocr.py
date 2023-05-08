import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import cv2
import os
from sklearn.model_selection import train_test_split


def get_data_ocr():
    """
    Returns the data split into the proper train/test/val split.
    """
    processed_data_path = "./../data/ocr_data/"
    captcha = []
    captcha_labels = []
    for image_path in sorted(os.listdir(processed_data_path)):
        image = cv2.imread(processed_data_path + image_path, cv2.IMREAD_GRAYSCALE)

        label = image_path.split('.')[0]
        captcha.append(image)
        captcha_labels.append(list(label))

    X_train, X_testval, Y_train, Y_testval = train_test_split(np.asarray(captcha, dtype=object), np.asarray(captcha_labels, dtype=object), test_size=.3, random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_testval, Y_testval, test_size=0.5, random_state=42)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val


def split_it_up_ocr():
    """
    Gets processed data from original folder and splits it into new folders train, test, and val
    """
    save_folder = "./../data/ocr_data_split/"
    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data_ocr()
    for img, label in zip(X_train, Y_train):
        item_name = ''.join(label)
        cv2.imwrite(save_folder + "train/" + item_name + '.png', np.asarray(img, dtype=np.uint8))

    for img, label in zip(X_test, Y_test):
        item_name = ''.join(label)
        cv2.imwrite(save_folder + "test/" + item_name + '.png', np.asarray(img, dtype=np.uint8))

    for img, label in zip(X_val, Y_val):
        item_name = ''.join(label)
        cv2.imwrite(save_folder + "val/" + item_name + '.png', np.asarray(img, dtype=np.uint8))


def retrieve_data_ocr(data_path):
    """
    Function to handle reshaping and formatting of data for OCR model.

    Inputs:
    test_split - Percent of data set asside for testing
    data_path - Path to data to be used

    Outputs:
    X_train, X_test, X_val, Y_train, Y_test, Y_val - Divided data (X) and labels (Y)
    char_encoder - Encoder used to ohe our data
    encoder_size - Number of ohe values used.
    """
    # Import and reshape
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_split_data_ocr(data_path)

    # Generate a one hot encoder for the dataset
    char_encoder = preprocessing.LabelEncoder().fit(np.concatenate((Y_train.reshape(-1), Y_test.reshape(-1), Y_val.reshape(-1))))
    orig_shapes  = [Y_train.shape[0], Y_test.shape[0], Y_val.shape[0]]

    # Encode from characters to numbers and reshape all of the labels
    Y_train = char_encoder.transform(Y_train.reshape(-1)).reshape((orig_shapes[0], 4))
    Y_test  = char_encoder.transform(Y_test.reshape(-1)).reshape((orig_shapes[1], 4))
    Y_val   = char_encoder.transform(Y_val.reshape(-1)).reshape((orig_shapes[2], 4))

    # Convert input to properly shaped and typed tensors
    X_train = tf.convert_to_tensor(np.asarray(X_train).astype(np.float32))
    Y_train = tf.convert_to_tensor(np.asarray(Y_train))
    X_val   = tf.convert_to_tensor(np.asarray(X_val).astype(np.float32))
    Y_val   = tf.convert_to_tensor(np.asarray(Y_val))

    # Don't reshape test, because we need to keep letters together as one CAPTCHA, but convert to tensor
    X_test  = tf.convert_to_tensor(np.asarray(X_test).astype(np.float32))
    Y_test  = tf.convert_to_tensor(np.asarray(Y_test))

    # Expand Dims
    X_train = tf.expand_dims(X_train, axis=-1)
    X_val   = tf.expand_dims(X_val  , axis=-1)
    X_test  = tf.expand_dims(X_test , axis=-1)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def get_split_data_ocr(processed_data_path):
    """
    Retrieves split data from their respective folders after preprocessing has been done

    processed_data_path - Path of preprocessed data
    """
    arrs = ([], [], [], [], [], [])
    for i, state in enumerate(["train/", "test/", "val/"]):
        for image_path in sorted(os.listdir(processed_data_path + state)):
            image = cv2.imread(processed_data_path + state + image_path, cv2.IMREAD_GRAYSCALE)

            label = image_path.split('.')[0]
            arrs[i*2].append(image)
            arrs[i*2 + 1].append(list(label))

    return np.asarray(arrs[0]), np.asarray(arrs[1]), np.asarray(arrs[2]), np.asarray(arrs[3]), np.asarray(arrs[4]), np.asarray(arrs[5])
