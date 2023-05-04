import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from preprocess import get_data


def retrieve_data(test_split, data_path):
    """
    Function to handle reshaping and formatting of data.

    Inputs:
    test_split - Percent of data set asside for testing
    data_path - Path to data to be used

    Outputs:
    X_train, X_test, X_val, Y_train, Y_test, Y_val - Divided data (X) and labels (Y)
    char_encoder - Encoder used to ohe our data
    encoder_size - Number of ohe values used.
    """
    # Import and reshape
    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data(test_split, data_path)

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

    return X_train, X_test, X_val, Y_train, Y_test, Y_val, char_encoder, encoder_size


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


def print_results(model, char_encoder, X_test, Y_test):
    """
    Prints an example CAPTCHA prediction vs. actual and prints accuracy

    Inputs:
    model - Model to get accuracy and example with
    char_encoder - one hot encoder used to labels of dataset
    X_test - CAPTCHA images to test with
    Y_test - True CAPTCHA labels to compare to
    """
    output = model.predict(X_test[6], verbose=0)
    prediction_captcha = char_encoder.inverse_transform(np.argmax(output, axis=1))
    real_captcha = char_encoder.inverse_transform(Y_test[6])

    # Convert accuracies to percentages
    char_acc, captcha_acc = get_accuracy(model, X_test, Y_test)
    char_acc = f"{round(100*char_acc, 2)}%"
    captcha_acc = f"{round(100*captcha_acc, 2)}%"

    print(f"Example CAPTCHA:     {real_captcha}")
    print(f"Example Model Guess: {prediction_captcha}")
    print(f"Character Accuracy:  {char_acc}")
    print(f"CAPTCHA Accuracy:    {captcha_acc}")


def main():
    X_train, X_test, X_val, Y_train, Y_test, Y_val, char_encoder, encoder_size = \
        retrieve_data(.3, "./processed_data/")

    input_shape=(X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])

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
    model.save('./../models/segmented_2', save_format="h5")

    print_results(model, char_encoder, X_test, Y_test)


if __name__ == '__main__':
    main()