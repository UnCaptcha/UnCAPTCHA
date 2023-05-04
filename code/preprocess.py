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

def get_data_ocr(test_split, processed_data_path):
    captcha = []
    captcha_labels = []
    count = 0
    for image_path in sorted(os.listdir(processed_data_path)):
        count += 1
        image = cv2.imread(processed_data_path + image_path, cv2.IMREAD_GRAYSCALE)

        label = image_path.split('.')[0]
        captcha.append(image)
        captcha_labels.append(list(label))



    X_train, X_testval, Y_train, Y_testval = train_test_split(np.asarray(captcha, dtype=object), np.asarray(captcha_labels, dtype=object), test_size=test_split)
    X_test, X_val, Y_test, Y_val = train_test_split(X_testval, Y_testval, test_size=0.5)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val


# save_folder = "./../split_processed_data/"
# X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data(0.3, "./../processed_data/")

# for imgs, labels in zip(X_train, Y_train):
#     for i, img in enumerate(imgs):
#         item_name = ''.join(labels) + str(i) + labels[i]
#         cv2.imwrite(save_folder + "train/" + item_name + '.png', np.asarray(img, dtype=np.uint8))

# for imgs, labels in zip(X_test, Y_test):
#     for i, img in enumerate(imgs):
#         item_name = ''.join(labels) + str(i) + labels[i]
#         cv2.imwrite(save_folder + "test/" + item_name + '.png', np.asarray(img, dtype=np.uint8))

# for imgs, labels in zip(X_val, Y_val):
#     for i, img in enumerate(imgs):
#         item_name = ''.join(labels) + str(i) + labels[i]
#         cv2.imwrite(save_folder + "val/" + item_name + '.png', np.asarray(img, dtype=np.uint8))


def get_split_data(processed_data_path):
    images = []
    labels = []
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

def get_split_data_ocr(processed_data_path):
    captcha = []
    captcha_labels = []
    arrs = ([], [], [], [], [], [])
    for i, state in enumerate(["train", "test", "val"]):
        for image_path in sorted(os.listdir(processed_data_path + state + "/")):
            image = cv2.imread(processed_data_path + state + "/" + image_path, cv2.IMREAD_GRAYSCALE)

            label = image_path.split('.')[0]
            arrs[i*2].append(image)
            arrs[i*2 + 1].append(list(label))

    return np.asarray(arrs[0]), np.asarray(arrs[1]), np.asarray(arrs[2]), np.asarray(arrs[3]), np.asarray(arrs[4]), np.asarray(arrs[5])

X_train, Y_train, X_test, Y_test, X_val, Y_val = get_split_data("./../split_processed_data/")
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)