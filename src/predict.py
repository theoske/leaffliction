import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import argparse
import re


CLASS_NAMES = [
    "Apple_Black_rot",
    "Apple_healthy",
    "Apple_rust",
    "Apple_scab",
    "Grape_Black_rot",
    "Grape_Esca",
    "Grape_healthy",
    "Grape_spot"
]

ORIGINAL_IMAGES_DIR = pathlib.Path("../images")


def find_original_image(variant_path: pathlib.Path) -> pathlib.Path:
    match = re.match(r"image\s+\((\d+)\)_(.+)\.JPG", variant_path.name, re.IGNORECASE)
    if not match:
        match = re.match(r"image\s+\((\d+)\)_(.+)", variant_path.name, re.IGNORECASE)
    if match:
        number = match.group(1)
        class_name = variant_path.parent.name
        original_name = f"image ({number}).JPG"
        original_path = ORIGINAL_IMAGES_DIR / class_name / original_name
        if original_path.exists():
            return original_path
        test_path = pathlib.Path("../test_images") / class_name / original_name
        if test_path.exists():
            return test_path
    return None


def predict(model_path: str, image_path: str):
    model_file = pathlib.Path(model_path)
    if not model_file.exists():
        print(f"ERROR: model {model_path} doesn't exist.")
        return
    image_file = pathlib.Path(image_path)
    if not image_file.exists():
        print(f"ERROR: image {image_path} doesn't exist.")
        print(f"Hint: Use quotes around the path: \"{image_path}\" or escape parentheses: image\(\d+\).JPG")
        return
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class_idx = tf.argmax(predictions[0]).numpy()
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = tf.reduce_max(predictions[0]).numpy()
    print(f"Predicted class: {predicted_class}")
    is_correct = predicted_class == image_file.parent.name
    bg_color = 'lightgreen' if is_correct else 'lightcoral'
    original_path = find_original_image(image_file)
    if original_path:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    fig.patch.set_facecolor(bg_color)
    if original_path:
        original_img = mpimg.imread(original_path)
        axes[0].imshow(original_img)
        axes[0].set_title(f"{original_path.parent.name}/{original_path.name}", fontsize=16)
        axes[0].axis('off')
        variant_img = mpimg.imread(image_file)
        axes[1].imshow(variant_img)
        axes[1].set_title(f"{image_file.parent.name}/{image_file.name}", fontsize=16)
        axes[1].axis('off')
        fig.text(0.5, 0.01, f"Predicted: {predicted_class}",
                 ha='center', fontsize=14, fontweight='bold')
    else:
        img_display = mpimg.imread(image_file)
        axes.imshow(img_display)
        axes.set_title(f"{image_file.parent.name}/{image_file.name}", fontsize=16)
        axes.axis('off')
        fig.text(0.5, 0.01, f"Predicted: {predicted_class}",
                 ha='center', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model.keras file")
    parser.add_argument("image_path", help="Path to image file")
    args = parser.parse_args()
    predict(args.model_path, args.image_path)


if __name__ == "__main__":
    main()
