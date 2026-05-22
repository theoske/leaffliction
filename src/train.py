import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pathlib
import argparse
from shutil import make_archive, rmtree, copy
import os

from Augmentation import balance_dataset
from Transformation import process_base_directory

NEWDATA_DIR = "./newdata"

print(tf.config.list_physical_devices())


def get_data(data_dir: str, test_dir: str = f"{NEWDATA_DIR}/test"):
    """
    Takes images from folder, divides them into training, evaluating and
    testing datasets and returns them in keras format.
    """
    if not os.path.isdir(data_dir):
        print("ERROR: No data for training.")
        exit(-1)
    print(data_dir)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg'))) + len(list(data_dir.glob('*/*.JPG')))
    print(image_count)
    if image_count < 1:
        print("ERROR: No data for training.")
        exit(-1)
    training_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(256, 256),
        batch_size=32,
        label_mode="categorical"
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(256, 256),
        batch_size=32,
        label_mode="categorical"
    )
    class_names = training_data.class_names
    testing_data = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=42,
        image_size=(256, 256),
        batch_size=32,
        label_mode="categorical"
    )
    print(class_names)
    return training_data, validation_data, testing_data


def display_training(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()


def prepare_ds(base_img: str):
    """
    calls augmentation and transformation for them to create :
        - augmented_images/
        - test_images/
        - transformed_images/
    """
    if not os.path.isdir(base_img):
        print(f"ERROR: \'{base_img}\' is not a directory.")
        exit(-1)
    test_img = "test"
    aug_img = f"{NEWDATA_DIR}/augmented_images"
    trans_img = f"{NEWDATA_DIR}/transformed_images"
    if os.path.exists(NEWDATA_DIR):
        print(f"INFO: {NEWDATA_DIR} already exists, skipping augmentation and transformation.")
        return trans_img
    balance_dataset(base_img, aug_img, test_path=test_img)
    process_base_directory(aug_img, trans_img)
    return trans_img


def archive_training(data_path: str = NEWDATA_DIR,
                     model_path: str = "model.keras", dst: str = "archive"):
    """
    Saves training datasets and resulting model in a zip file.
    """
    if not os.path.exists(data_path):
        print(f"ERROR: can't zip unexistant folder {data_path}")
        return
    if not os.path.exists(model_path):
        print(f"ERROR: can't zip unexistant model {model_path}")
        return
    if os.path.exists(dst+".zip"):
        print(f"WARNING: {dst}.zip folder already exists. Deleting it...")
        os.remove(dst+".zip")
    copy(model_path, data_path)
    make_archive(dst, "zip", data_path)


def train(data_dir: str):
    """
    Core of the CNN.
    Makes the data go through 4 blocks of convolutions and feeds the resulting
    data to the 2 layer neural network.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Active l'allocation dynamique de la mémoire (Memory Growth)
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Optimisation VRAM : Activée")
        except RuntimeError as e:
            print(e)
    data_dir = prepare_ds(data_dir)
    training_ds, validation_ds, testing_ds = get_data(data_dir)
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(256, 256, 3)),
        # Block 1
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Block 2
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Block 3
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Block 4
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(8, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5,
                                                  restore_best_weights=True)
    history = model.fit(training_ds, epochs=10, validation_data=validation_ds,
                        callbacks=[early_stop])
    test_loss, test_acc = model.evaluate(testing_ds)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    model.save("model.keras", overwrite=True)
    archive_training(NEWDATA_DIR)
    display_training(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?",
                        help="Path to ds", default="../images")
    args = parser.parse_args()
    train(args.data_dir)


if __name__ == "__main__":
    main()
