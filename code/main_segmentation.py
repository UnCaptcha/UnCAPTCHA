import numpy as np
import tensorflow as tf
from get_data_segmentation import retrieve_data


def create_model(input_shape, encoder_size):
    """
    Helper function to instantiate a version of our model with correct inputs

    Inputs:
    input_shape - shape of the input CAPTCHAs
    encoder_size - Number of unique characters that can be predicted

    Output:
    model - returns our model to be used
    """
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
            tf.keras.layers.Dense(encoder_size, activation="softmax")
        ])
    return model


def get_accuracy(model, X_test, Y_test):
    """
    Calculates model accuracy

    Inputs:
    model - Model you are testing accuracy on
    X_test - CAPTCHAs to test accuracy on
    Y_test - True text for each input CAPTCHA

    Output:
    char_acc - Percentage of the time the model correctly predicts a character
    captcha_acc - Percentage of the time the model correctly predicts an entire CAPTCHA
    """
    # Flattens the data
    X_test = np.reshape(X_test, (-1, *X_test.shape[-2:]))
    Y_test = np.reshape(Y_test, (-1))

    # Compares every character prediction to true label and leaves accuracy an array of 1s and 0s
    # representing whether or not it correctly predicted that character.
    probs       = model.predict(X_test, verbose=0)
    diff        = np.argmax(probs, axis=1) - Y_test
    accuracy    = np.where(diff == 0, 1, 0)

    # Compute character and captcha accuracy from above results
    char_acc    = np.mean(accuracy)
    captcha_acc = np.array_split(diff, diff.shape[0]//4)
    captcha_acc = np.apply_along_axis((lambda x: 1 if np.all(x == 0) else 0), axis=1, arr=captcha_acc)
    captcha_acc = np.mean(captcha_acc)

    return char_acc, captcha_acc


def print_results(model, X_test, Y_test):
    """
    Prints an example CAPTCHA prediction vs. actual and prints accuracy

    Inputs:
    model - Model to get accuracy and example with
    char_encoder - one hot encoder used to labels of dataset
    X_test - CAPTCHA images to test with
    Y_test - True CAPTCHA labels to compare to
    """
    output = model.predict(X_test[6], verbose=0)
    alphabet_key = dict(zip(range(0, 33), list('23456789ABCDEFGHJKLMNPQRSTUVWXYZ_')))
    prediction_captcha = [alphabet_key[i] for i in np.argmax(output, axis=1)]
    real_captcha = [alphabet_key[i] for i in np.asarray(Y_test[6])]

    # Convert accuracies to percentages
    char_acc, captcha_acc = get_accuracy(model, X_test, Y_test)
    char_acc = f"{round(100*char_acc, 2)}%"
    captcha_acc = f"{round(100*captcha_acc, 2)}%"

    print(f"Example CAPTCHA:     {real_captcha}")
    print(f"Example Model Guess: {prediction_captcha}")
    print(f"Character Accuracy:  {char_acc}")
    print(f"CAPTCHA Accuracy:    {captcha_acc}")


def main():
    encoder_size = 31
    X_train, Y_train, X_test, Y_test, X_val, Y_val = \
        retrieve_data("./../data/segmented_data_split/")

    input_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])

    # Fit model
    model = create_model(input_shape, encoder_size)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(.0001))
    model.fit(
        X_train,
        Y_train,
        epochs=3,
        batch_size=265,
        validation_data=(X_val, Y_val)
    )

    # Save model for future testing
    model.save('./../models/segmented', save_format="h5")

    model = tf.keras.models.load_model("./../models/segmented")

    print_results(model, X_test, Y_test)


if __name__ == '__main__':
    main()