import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pathlib
import argparse

print(tf.config.list_physical_devices())

def get_data(data_dir: str, test_dir: str = "../test_images"):
    print(data_dir)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
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
    training_d2 = tf.keras.utils.image_dataset_from_directory(
        '../augmented_images',
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(256, 256),
        batch_size=32,
        label_mode="categorical"
    )
    validation_d2 = tf.keras.utils.image_dataset_from_directory(
        '../augmented_images',
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(256, 256),
        batch_size=32,
        label_mode="categorical"
    )
    class_names = training_data.class_names
    training_data = training_data.concatenate(training_d2)
    validation_data = validation_data.concatenate(validation_d2)
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

def train(data_dir: str):
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
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(training_ds, epochs=10, validation_data=validation_ds, callbacks=[early_stop])
    test_loss, test_acc = model.evaluate(testing_ds)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    model.save("model.keras", overwrite=True)
    display_training(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?",
                        help="Path to ds", default="../transformed_images")
    args = parser.parse_args()
    train(args.data_dir)


if __name__ == "__main__":
    main()

