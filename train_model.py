"""
Cat Classifier Training Script
------------------------------
This script mounts Google Drive in Colab, loads a dataset of cat images,
trains a binary image classifier for a specific cat against other cats
using MobileNetV2 as base model, and saves the trained model. It also
displays sample images before training and evaluates the trained model.
"""

# Standard library
import os

# Third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from google.colab import drive



# -------------------------------------------------------------------
#   Constants
# -------------------------------------------------------------------

IMG_SIZE = (180, 180)
IMG_SIZE_ = 180
BATCH_SIZE = 16
CAT = "bruno"
DATASET_NAME = f"dataset_{CAT[:3]}_other"
VERSION = "4"

CLASS_0_DIR = f"/content/drive/MyDrive/datasets/{DATASET_NAME}/preprocessed_images/{CAT}"
CLASS_1_DIR = f"/content/drive/MyDrive/datasets/{DATASET_NAME}/preprocessed_images/other_cats"



# -------------------------------------------------------------------
#   Functions
# -------------------------------------------------------------------

def mount_google_drive():
    """Mount Google Drive in Colab to access dataset files."""
    drive.mount("/content/drive")


def load_image_paths():
    """
    Load image file paths and create a DataFrame with labels.

    Returns:
        pd.DataFrame: DataFrame containing image filenames and class labels.
    """
    class_0_images = [
        os.path.join(CLASS_0_DIR, fname)
        for fname in os.listdir(CLASS_0_DIR)
    ]
    class_1_images = [
        os.path.join(CLASS_1_DIR, fname)
        for fname in os.listdir(CLASS_1_DIR)
    ]

    df = pd.DataFrame({
        "filename": class_0_images + class_1_images,
        "class": [0] * len(class_0_images) + [1] * len(class_1_images)
    })
    df['class'] = df['class'].astype(str)
    return df


def create_generators(df):
    """
    Create training and validation generators using ImageDataGenerator.
    Applies rescaling and data augmentation for training.

    Args:
        df (pd.DataFrame): DataFrame containing filenames and labels.

    Returns:
        tuple: (train_generator, validation_generator)
    """
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.26,
        height_shift_range=0.26,
        shear_range=0.17,
        zoom_range=0.26,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=True,
        seed=42
    )

    return train_gen, val_gen


def display_sample_images(generator):
    """
    Display a batch of images from the generator with labels.

    Args:
        generator (keras.preprocessing.image.DataFrameIterator): 
            Image data generator.
    """
    images, labels = next(generator)
    plt.figure(figsize=(10, 10))
    for i in range(min(9, BATCH_SIZE)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {int(labels[i])}")
        plt.axis("off")
    plt.show()


def build_model():
    """
    Build and compile a MobileNetV2-based binary classifier.

    Returns:
        keras.Model: Compiled Keras model.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE_, IMG_SIZE_, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, train_gen, val_gen, cat_name, version):
    """
    Train a binary image classifier using the training generator and
    evaluate it on the validation generator. Saves the trained model
    to Google Drive.

    Args:
        model (keras.Model): Compiled Keras model.
        train_gen (keras.preprocessing.image.DataFrameIterator): Training data generator.
        val_gen (keras.preprocessing.image.DataFrameIterator): Validation data generator.
        cat_name (str): Name of the target cat.
        version (str): Version string for the saved model filename.

    Returns:
        str: Path to the saved model file.
    """
    model.fit(train_gen, validation_data=val_gen, epochs=5)

    model_path = f"/content/drive/MyDrive/repos/keras-distinguish-own-cat-from-others-model/models/model{version}_{cat_name}.keras"
    model.save(model_path, save_format="keras")
    return model_path


def evaluate_model(model_path, val_gen):
    """
    Load a saved model and evaluate its performance on validation data.

    Args:
        model_path (str): Path to the saved Keras model.
        val_gen (keras.preprocessing.image.DataFrameIterator): Validation data generator.
    """
    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(val_gen)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")



# -------------------------------------------------------------------
#   Main
# -------------------------------------------------------------------

def main():
    """Main function to run the full Google Colab training pipeline."""
    mount_google_drive()

    df = load_image_paths()
    train_gen, val_gen = create_generators(df)

    print(f"Class names: {CAT} = 0, other_cats = 1")
    display_sample_images(train_gen)

    model = build_model()
    model_path = train_model(model, train_gen, val_gen, CAT, VERSION)
    evaluate_model(model_path, val_gen)


if __name__ == "__main__":
    main()
