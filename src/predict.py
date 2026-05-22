import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory

IMAGES_DIR = "./leaves/images"


def load_and_preprocess_image(img_path):
    """
    Load and preprocess an image for prediction.
    np.expand_dims() is necessary because keras needs the channel dimension
    even for a single image.
    """
    try:
        img = image.load_img(img_path, target_size=(256, 256))
    except Exception as e:
        print(f"ERROR: {e}")
        exit(-1)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def get_pure_img(base_img_path: str):
    """
    Takes a modified image's path and returns the original, unmodified
    image using the image modification naming convention.
    """
    if not os.path.exists(IMAGES_DIR):
        print(f"no {IMAGES_DIR} folder")
        return None
    # gets only the file without the parent folders
    filename = base_img_path.split('/')[-1]
    foldername = base_img_path.split('/')[-2]
    # get the base filename without the modifiers
    pure = filename.split('_')[0]
    if ".JPG" not in pure:
        pure += ".JPG"
    pure_path = f"{IMAGES_DIR}/{foldername}/{pure}"
    if not os.path.exists(pure_path):
        print(f"{IMAGES_DIR}/{pure} does not exist")
        return None
    pure_img = image.load_img(pure_path, target_size=(256, 256))
    return pure_img


def evaluate_model(model_path, folder_path):
    """Evaluate the model on all JPG files in the given folder."""
    try:
        model = load_model(model_path)
    except ValueError as e:
        print(f"ERROR : {e}")
        exit(-1)
    print(f"Model loaded from {model_path}")
    try:
        dataset = image_dataset_from_directory(
            folder_path,
            image_size=(256, 256),
            batch_size=32,
            label_mode="categorical",
            shuffle=False
        )
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        exit(-1)
    try:
        model.evaluate(dataset)
    except Exception as e:
        print(f"ERROR : {e}")
        exit(-1)


def predict_and_display(model_path, img_path):
    """Load model, predict, and display result."""
    # Load the model
    try:
        model = load_model(model_path)
    except ValueError as e:
        print(f"ERROR : {e}")
        exit(-1)
    print(f"Model loaded from {model_path}")
    # Load and preprocess the image
    img, img_array = load_and_preprocess_image(img_path)
    print(f"Image loaded from {img_path}")
    # Make prediction
    class_names = [
        "Apple_Black_rot",
        "Apple_healthy",
        "Apple_rust",
        "Apple_scab",
        "Grape_Black_rot",
        "Grape_Esca",
        "Grape_healthy",
        "Grape_spot"
    ]
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    pure_img = get_pure_img(img_path)
    plt.figure(figsize=(10, 5), num=f"Prediction for {img_path.split('/')[-2]}\
               /{img_path.split('/')[-1]}")
    if pure_img is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(pure_img)
        plt.axis('off')
        plt.title("Pure Image")
        plt.subplot(1, 2, 2 if pure_img is not None else 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: Class {predicted_class} (Confidence: \
              {confidence:.2f})")
    plt.tight_layout()
    plt.savefig('prediction.png')
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Make a prediction using a \
                                     Keras model.")
    parser.add_argument("model_path", type=str, help="Path to the .keras \
                        model file")
    parser.add_argument("image_path", type=str, help="Path to the image file \
                        or folder")
    args = parser.parse_args()
    if os.path.isdir(args.image_path):
        evaluate_model(args.model_path, args.image_path)
    else:
        predict_and_display(args.model_path, args.image_path)


if __name__ == "__main__":
    main()
