import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def get_data(test_split, processed_data_path):
    images = []
    labels = []
    captcha = []
    captcha_labels = []
    count = 0
    for image_path in sorted(os.listdir(processed_data_path)):
        count += 1
        image = cv2.imread(processed_data_path + image_path, cv2.IMREAD_GRAYSCALE)

        label = image_path.split('.')[0][-1]
        captcha.append(image)
        captcha_labels.append(label)

        if count % 4 == 0:
            images.append(captcha)
            labels.append(captcha_labels)
            captcha = []
            captcha_labels = []

    X_train, X_testval, Y_train, Y_testval = train_test_split(np.asarray(images, dtype=object), np.asarray(labels, dtype=object), test_size=test_split)
    X_test, X_val, Y_test, Y_val = train_test_split(X_testval, Y_testval, test_size=0.5)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val

X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data(0.3, "./../processed_data/")
