import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_probability as tfp
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2

def build_bayesian_cnn_model(img_height, img_width):
    tfpl = tfp.layers
    tfd = tfp.distributions

    divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / train_generator.samples

    model_bayes = Sequential([
        tfpl.Convolution2DReparameterization(input_shape=(img_height, img_width, 3),
                                              filters=8, kernel_size=16, activation='relu',
                                              kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                                              kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                              kernel_divergence_fn=divergence_fn,
                                              bias_prior_fn=tfpl.default_multivariate_normal_fn,
                                              bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                              bias_divergence_fn=divergence_fn),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.28),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.28),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.28),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.28),
        tfpl.DenseReparameterization(units=tfpl.OneHotCategorical.params_size(2), activation=None,
                                     kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                                     kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                     kernel_divergence_fn=divergence_fn,
                                     bias_prior_fn=tfpl.default_multivariate_normal_fn,
                                     bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                     bias_divergence_fn=divergence_fn
                                    ),
        tfpl.OneHotCategorical(2)
    ])
    
    return model_bayes

def negative_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def train_bayesian_cnn_model(model, train_generator, validation_generator, epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0025),
                  loss=negative_log_likelihood,
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        verbose=1
    )

    return history

def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def load_and_preprocess_image(image_path, img_height, img_width):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (img_height, img_width))
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)
    return img_array

def get_image_probabilities(model, img_array):
    predictions = model.predict(img_array)
    probs = predictions.mean().numpy()[0]
    return probs

def plot_image_probabilities(model, img_array):
    predictions = model.predict(img_array)
    probs = predictions.mean().numpy()[0]
    
    plt.figure(figsize=(8, 4))
    plt.bar([0, 1], probs, color=['blue', 'red'])
    plt.xticks([0, 1], ['Not Infected', 'Infected'])
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.show()


if __name__ == "__main__":
    img_height, img_width = 256, 256

    data_dir = 'C:\\Users\\Dell\\Desktop\\segmented_split'

    batch_size = 32
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
        vertical_flip=False,
        rotation_range=20,
        zoom_range=0.1,
        shear_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    model_bayes = build_bayesian_cnn_model(img_height, img_width)
    epochs = 10
    history = train_bayesian_cnn_model(model_bayes, train_generator, validation_generator, epochs)
    plot_training_history(history)

   
