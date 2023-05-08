import numpy as np
import tensorflow as tf
from preprocess import retrieve_data_ocr


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
            #CNN layers of our CRNN
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
            tf.keras.layers.MaxPool2D(pool_size=(2,3)),

            tf.keras.layers.Reshape((input_shape[-2] // 12, input_shape[-3] // 8 * 128)),

            #RNN segment of our CRNN
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_shape[-3] // 8 * 128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_shape[-3] // 8 * 128, return_sequences=True)),

            # +1 IS NECESSARY for CTC loss.
            tf.keras.layers.Dense(encoder_size+1, activation="softmax")
        ])
    return model


def ctc(y_true, y_pred):
    """
    A custom loss function built off of Keras' ctc_batch_cost which calculates loss for our CRNN

    Inputs:
    y_true - True labels
    y_pred - Predicted labels

    Output:
    loss - Loss as calculated by ctc_batch_cost
    """
    # Cast true labels to ints
    y_true = tf.cast(y_true, tf.int32)

    # Calculate label_length and input_length sizes for ctc_batch_cost by appending batch size
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_len = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_len = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_len * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_len * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, label_length=label_length, input_length=input_length)

    return loss


def get_accuracy(model, X_test, Y_test):
    """
    Calculates model accuracy

    Inputs:
    model - Model you are testing accuracy on
    X_test - CAPTCHAs to test accuracy on
    Y_test - True text for each input CAPTCHA

    Output:
    accuracy - Percentage of the time the model correctly reads the entire CAPTCHA
    """
    # Use model to predict text
    probs = model.predict(X_test, verbose=0)
    output = np.argmax(probs, axis=2)

    # The output is from an LSTM which looks something like [1,1,31,25,7,7]
    # This takes it down to the four unique characters predicted in a row
    prediction = np.asarray([np.asarray(tf.unique_with_counts(x)[0]) for x in output])
    Y_test = np.asarray(Y_test)

    # Calculate accuracy by comparing predictions with truth.
    count = 0
    total = Y_test.shape[0]
    for x, y in zip(prediction, Y_test):
        if len(x) == len(y) and np.all(np.equal(x, y)): count += 1
    accuracy = count / total

    return accuracy


def print_results(model, X_test, Y_test):
    """
    Prints an example CAPTCHA prediction vs. actual and prints accuracy

    Inputs:
    model - Model to get accuracy and example with
    X_test - CAPTCHA images to test with
    Y_test - True CAPTCHA labels to compare to
    """
    output = model.predict(X_test[1:2], verbose=0)
    output = np.transpose(output, axes=[1, 0, 2]).squeeze()
    output= np.argmax(output, axis=1)
    prediction_captcha = np.asarray([np.asarray(tf.unique_with_counts(x)[0]) for x in output])

    # Decode label's one hot encoding
    alphabet_key = dict(zip(range(0, 33), list('23456789ABCDEFGHJKLMNPQRSTUVWXYZ_')))
    real_captcha = [alphabet_key[i] for i in np.asarray(Y_test[1])]

    # Get accuracy
    accuracy = get_accuracy(model, X_test, Y_test)

    #Print example CAPTCHA and accuracy
    print(f"Example CAPTCHA:     {real_captcha}")
    print(f"Example Model Guess: {prediction_captcha}")
    print(f"Accuracy: {accuracy}")


def main():
    encoder_size = 31
    X_train, Y_train, X_test, Y_test, X_val, Y_val = \
        retrieve_data_ocr("./../data/split_processed_whole_data/")
    # Input shape is the shape of X_train without batch_size attached
    input_shape=(X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])

    # Fit model
    # NOTE: encoder_size incremented by 1 is necessary here AND inside the model for CTC_loss to work
    # Having it twice is NOT an error.
    model = create_model(input_shape, (encoder_size+1))
    model.compile(loss=ctc,
                    optimizer=tf.keras.optimizers.Adam(.0001))
    model.fit(
        X_train,
        Y_train,
        epochs=80,
        batch_size=256,
        validation_data=(X_val, Y_val)
    )

    # Save model for future testing
    model.save('./models/ocr', save_format="h5")

    print_results(model, X_test, Y_test)


if __name__ == '__main__':
    main()