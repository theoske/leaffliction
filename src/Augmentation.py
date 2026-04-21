import os
import shutil
import argparse
import random
import numpy as np
import cv2
from glob import glob
from Distribution import get_distribution
from global_var import *


def apply_flip(image: np.ndarray) -> np.ndarray:
    """Horizontal flip."""
    return cv2.flip(image, 1)


def apply_rotate(image: np.ndarray) -> np.ndarray:
    """Random rotation between -30 and 30 degrees."""
    h, w = image.shape[:2]
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(image, M, (new_w, new_h),
                          borderMode=cv2.BORDER_REFLECT_101)


def apply_skew(image: np.ndarray) -> np.ndarray:
    """Perspective (projective) transformation."""
    h, w = image.shape[:2]
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [np.random.randint(0, margin_x), np.random.randint(0, margin_y)],
        [w - np.random.randint(0, margin_x), np.random.randint(0, margin_y)],
        [w - np.random.randint(0, margin_x), h - np.random.randint(0, margin_y)],
        [np.random.randint(0, margin_x), h - np.random.randint(0, margin_y)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)


def apply_shear(image: np.ndarray) -> np.ndarray:
    """Affine shear transformation."""
    h, w = image.shape[:2]
    shear = np.random.uniform(-0.3, 0.3)
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    new_w = int(w + abs(shear) * h)
    if shear < 0:
        M[0, 2] = abs(shear) * h
    return cv2.warpAffine(image, M, (new_w, h),
                          borderMode=cv2.BORDER_REFLECT_101)


def apply_crop(image: np.ndarray) -> np.ndarray:
    """Zoom into center of image and resize to original dimensions."""
    h, w = image.shape[:2]
    scale = np.random.uniform(1.2, 1.6)
    crop_h, crop_w = int(h / scale), int(w / scale)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_distortion(image: np.ndarray) -> np.ndarray:
    """Barrel / pincushion lens distortion."""
    h, w = image.shape[:2]
    cam = np.array([[w, 0, w / 2],
                    [0, h, h / 2],
                    [0, 0, 1]], dtype=np.float32)
    k1 = np.random.uniform(-0.4, -0.1)
    dist_coeffs = np.array([k1, 0, 0, 0], dtype=np.float32)
    return cv2.undistort(image, cam, dist_coeffs)


TRANSFORM_FUNCS = {
    "Flip": apply_flip,
    "Rotate": apply_rotate,
    "Skew": apply_skew,
    "Shear": apply_shear,
    "Crop": apply_crop,
    "Distortion": apply_distortion,
}

TRANSFORMATIONS = list(TRANSFORM_FUNCS.keys())
RANDOM_TRANSFORMATIONS = [k for k in TRANSFORM_FUNCS if k != "Flip"]


def augment_image(image_path: str, output_dir: str = None):
    """Apply all 6 augmentations to a single image and save results."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read {image_path}")
        return []

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(image_path))

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    saved = []

    for name, func in TRANSFORM_FUNCS.items():
        augmented = func(image)
        out_path = os.path.join(output_dir, f"{base}_{name}{ext}")
        cv2.imwrite(out_path, augmented)
        saved.append(out_path)
        print(f"  {base}_{name}{ext}")

    return saved


def select_categories_to_augment(data_path: str = IMAGES_PATH):
    """Find under-represented categories and return them with target size."""
    distrib = get_distribution(data_path)

    max_size = 0
    for category in distrib:
        for d in distrib[category]:
            if d["size"] > max_size:
                max_size = d["size"]

    to_augment = []
    for category in distrib:
        for d in distrib[category]:
            if d["size"] < max_size:
                to_augment.append((d["name"], d["size"], max_size))

    return to_augment


def split_test(data_path: str, test_ratio: float = 0.2,
                     min_test: int = 100, seed: int = 42):
    """
    Move test_ratio of images from each category into an
    'test_images' folder (sibling of data_path).
    Returns the path to the test directory.
    """
    test_path = os.path.join(os.path.dirname(data_path), "test_images")
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    rng = random.Random(seed)
    for category in sorted(os.listdir(data_path)):
        cat_dir = os.path.join(data_path, category)
        if not os.path.isdir(cat_dir):
            continue
        images = glob(os.path.join(cat_dir, "*.JPG"))
        if not images:
            continue
        images.sort()
        n_test = max(min_test, int(len(images) * test_ratio))
        rng.shuffle(images)
        test_images = images[:n_test]
        test_cat_dir = os.path.join(test_path, category)
        os.makedirs(test_cat_dir, exist_ok=True)
        for img_path in test_images:
            shutil.move(img_path, test_cat_dir)
        print(f"{category}: {n_test}/{len(images)} images -> test")
    print(f"\ntest set saved to: {test_path}\n")
    return test_path


def balance_dataset(data_path: str = IMAGES_PATH,
                    output_path: str = None):
    """
    Copy the dataset to augmented_images, split 20% of each category
    into test_images, then augment under-represented categories
    to balance the training set.
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(data_path), "augmented_images")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    shutil.copytree(data_path, output_path)
    print(f"Copied dataset to {output_path}")
    split_test(output_path)
    to_augment = select_categories_to_augment(output_path)
    if not to_augment:
        print("Dataset is already balanced.")
        return
    for name, current_size, target_size in to_augment:
        dir_path = os.path.join(output_path, name)
        images = glob(os.path.join(dir_path, "*.JPG"))
        if not images:
            continue
        needed = target_size - current_size
        generated = 0
        flipped = set()
        while generated < needed:
            img_path = images[generated % len(images)]
            image = cv2.imread(img_path)
            if image is None:
                generated += 1
                continue
            base = os.path.splitext(os.path.basename(img_path))[0]
            ext_out = os.path.splitext(img_path)[1]
            if img_path not in flipped:
                tf_name = "Flip"
                flipped.add(img_path)
            else:
                tf_name = RANDOM_TRANSFORMATIONS[
                    generated % len(RANDOM_TRANSFORMATIONS)
                ]
            augmented = TRANSFORM_FUNCS[tf_name](image)
            out_path = os.path.join(dir_path,
                                    f"{base}_aug{generated}_{tf_name}{ext_out}")
            cv2.imwrite(out_path, augmented)
            generated += 1
        print(f"{name}: +{generated} images ({current_size} -> {target_size})")
    print(f"\nBalanced dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Image data augmentation: Flip, Rotate, Skew, "
                    "Shear, Crop, Distortion"
    )
    parser.add_argument("image", nargs="?",
                        help="Path to an image to augment")
    parser.add_argument("-b", "--balance", action="store_true",
                        help="Balance the dataset by augmenting "
                             "under-represented categories")
    parser.add_argument("-o", "--output",
                        help="Output directory")
    args = parser.parse_args()
    if args.balance:
        balance_dataset(IMAGES_PATH, args.output)
    elif args.image:
        print(f"Augmenting: {args.image}")
        saved = augment_image(args.image, args.output)
        if saved:
            print(f"Saved {len(saved)} augmented images.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
