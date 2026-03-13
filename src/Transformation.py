import os
import shutil
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from plantcv import plantcv as pcv


TRANSFORMS = [
    "gaussian_blur",
    "mask",
    "roi_objects",
    "analyze_object",
    "pseudolandmarks",
]


def transform_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to the image."""
    return pcv.gaussian_blur(img=image, ksize=(15, 15), sigma_x=0)


def transform_mask(image: np.ndarray) -> np.ndarray:
    """Generate a binary mask of the leaf."""
    gray = pcv.rgb2gray_lab(rgb_img=image, channel="a")
    mask = pcv.threshold.binary(
        gray_img=gray, threshold=115, object_type="dark"
    )
    mask = pcv.fill(bin_img=mask, size=300)
    mask = pcv.closing(gray_img=mask)
    mask = pcv.erode(gray_img=mask, ksize=3, i=1)
    mask = pcv.dilate(gray_img=mask, ksize=3, i=1)
    return mask


def transform_roi_objects(image: np.ndarray) -> np.ndarray:
    """Show leaf region of interest: green overlay on leaf, blue bounding box."""
    mask = transform_mask(image)
    h, w = image.shape[:2]

    roi = pcv.roi.rectangle(img=image, x=0, y=0, h=h, w=w)
    filtered_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="largest")

    gray_bg = cv2.cvtColor(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
    )
    green_overlay = gray_bg.copy()
    green_overlay[filtered_mask > 0] = (0, 255, 0)

    contours, _ = cv2.findContours(
        filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(green_overlay, (x, y), (x + bw, y + bh), (255, 0, 0), 3)

    return green_overlay


def transform_analyze_object(image: np.ndarray) -> np.ndarray:
    """Analyze leaf shape using PlantCV's size analysis."""
    mask = transform_mask(image)
    labeled_mask = np.where(mask > 0, 1, 0).astype(np.int32)
    return pcv.analyze.size(img=image, labeled_mask=labeled_mask)


def transform_pseudolandmarks(image: np.ndarray) -> np.ndarray:
    """Place pseudolandmarks along the leaf contour using PlantCV."""
    mask = transform_mask(image)
    result = image.copy()

    top, bottom, center = pcv.homology.x_axis_pseudolandmarks(
        img=image, mask=mask
    )

    if not isinstance(top, np.ndarray):
        return result

    groups = [
        (top, (255, 0, 0)),
        (bottom, (0, 128, 255)),
        (center, (255, 0, 255)),
    ]

    for pts, color in groups:
        for pt in pts:
            coord = (int(pt[0][0]), int(pt[0][1]))
            cv2.circle(result, coord, 5, color, -1)

    return result


TRANSFORM_DISPLAY = {
    "gaussian_blur": ("Gaussian blur", transform_gaussian_blur),
    "mask": ("Mask", transform_mask),
    "roi_objects": ("ROI objects", transform_roi_objects),
    "analyze_object": ("Analyze object", transform_analyze_object),
    "pseudolandmarks": ("Pseudolandmarks", transform_pseudolandmarks),
}


def display_transformations(image_path: str):
    """Display all transformations for a single image using matplotlib."""
    image = cv2.imread(image_path)
    if image is None:
        return

    pcv.params.debug = None

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.canvas.manager.set_window_title("Transformation.py")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transforms_to_show = [
        ("Original", rgb),
        ("Gaussian blur", cv2.cvtColor(
            transform_gaussian_blur(image), cv2.COLOR_BGR2RGB)),
        ("Mask", transform_mask(image)),
        ("ROI objects", cv2.cvtColor(
            transform_roi_objects(image), cv2.COLOR_BGR2RGB)),
        ("Analyze object", cv2.cvtColor(
            transform_analyze_object(image), cv2.COLOR_BGR2RGB)),
        ("Pseudolandmarks", cv2.cvtColor(
            transform_pseudolandmarks(image), cv2.COLOR_BGR2RGB)),
    ]

    for ax, (title, img) in zip(axes.flat, transforms_to_show):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def save_transform(image_path: str, transform_name: str, dst_dir: str):
    """Apply one transform to an image and save the result."""
    image = cv2.imread(image_path)
    if image is None:
        return

    pcv.params.debug = None

    base = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]

    _, func = TRANSFORM_DISPLAY[transform_name]
    result = func(image)
    out_path = os.path.join(dst_dir, f"{base}_{transform_name}{ext}")
    cv2.imwrite(out_path, result)

    return out_path


def process_directory(src_dir: str, dst_dir: str, transform_name: str):
    """Apply a transform to all images in a directory and save results.

    Collects image paths by joining src_dir with each glob extension pattern
    (e.g. "*.JPG"), expanding matches, and extending the images list across
    all supported extensions.
    """
    os.makedirs(dst_dir, exist_ok=True)

    extensions = ("*.JPG", "*.jpg", "*.png", "*.PNG", "*.jpeg", "*.JPEG")
    images = []
    for ext in extensions:
        images.extend(glob(os.path.join(src_dir, ext)))

    if not images:
        return

    for img_path in images:
        out = save_transform(img_path, transform_name, dst_dir)

    print(f"Processed {len(images)} images -> {dst_dir}")


def process_base_directory(base_dir: str):
    """Apply all transforms to all images in all subdirectories.

    Creates a 'transformed_directory' sibling folder preserving
    the full subdirectory structure of base_dir.
    """
    base_dir = os.path.abspath(base_dir)
    dst_base = os.path.join(os.path.dirname(base_dir),
                            "transformed_directory")

    if os.path.exists(dst_base):
        shutil.rmtree(dst_base)

    for dirpath, _, _ in sorted(os.walk(base_dir)):
        rel_path = os.path.relpath(dirpath, base_dir)
        if rel_path == ".":
            continue
        dst_dir = os.path.join(dst_base, rel_path)
        for tf in TRANSFORMS:
            process_directory(dirpath, dst_dir, tf)


def main():
    parser = argparse.ArgumentParser(
        description="Leaf image transformation: Gaussian blur, mask, "
                    "ROI objects, analyze object, pseudolandmarks"
    )
    parser.add_argument("image", nargs="?",
                        help="Path to a single image (displays all transforms)")
    parser.add_argument("-s", "--source",
                        help="Source directory of images (batch mode)")
    parser.add_argument("-d", "--destination",
                        help="Destination directory for batch output")
    parser.add_argument("-a", "--all",
                        help="Base folder: apply all transforms to all "
                             "subdirectories and save to transformed_directory")

    for tf in TRANSFORMS:
        flag = tf.replace("_", "-")
        parser.add_argument(
            f"-{flag}", f"--{flag}",
            action="store_true",
            help=f"Apply {tf} transform"
        )

    args = parser.parse_args()

    if args.all:
        process_base_directory(args.all)
    elif args.image and not args.source:
        display_transformations(args.image)
    elif args.source and args.destination:
        selected = [
            tf for tf in TRANSFORMS
            if getattr(args, tf.replace("-", "_"), False)
        ]
        if not selected:
            selected = TRANSFORMS
        for tf in selected:
            process_directory(args.source, args.destination, tf)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
