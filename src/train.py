import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pathlib
import argparse


def get_data(data_dir: str, test_dir: str = "../evaluation_images"):
    print(data_dir)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    train_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(256, 256),
        batch_size=32,
        label_mode="categorical"
    )
    eval_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(256, 256),
        batch_size=32,
        label_mode="categorical"
    )
    test_data = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=42,
        image_size=(256, 256),
        batch_size=32,
        label_mode="categorical"
    )
    class_names = train_data.class_names
    print(class_names)
    return train_data, eval_data, test_data

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
    train_data, eval_data, test_data = get_data(data_dir)
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(256, 256, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(8, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(train_data, epochs=10, validation_data=eval_data)
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    model.save("model.keras", overwrite=True)
    display_training(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?",
                        help="Path to ds", default="../transformed_directory")
    args = parser.parse_args()
    train(args.data_dir)


if __name__ == "__main__":
    main()
