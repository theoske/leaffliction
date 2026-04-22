import os
import shutil
import argparse
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
    return pcv.gaussian_blur(img=image, ksize=(15, 15), sigma_x=0)


def transform_mask(image: np.ndarray) -> np.ndarray:
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
    mask = transform_mask(image)
    h, w = image.shape[:2]
    roi = pcv.roi.rectangle(img=image, x=0, y=0, w=w, h=h)
    filtered_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')
    return pcv.apply_mask(img=image, mask=filtered_mask, mask_color='black')


def transform_analyze_object(image: np.ndarray) -> np.ndarray:
    mask = transform_mask(image)
    labeled_mask = np.where(mask > 0, 1, 0).astype(np.int32)
    return pcv.analyze.size(img=image, labeled_mask=labeled_mask)


def transform_pseudolandmarks(image: np.ndarray) -> np.ndarray:
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
    yy, xx = np.ogrid[0:result.shape[0], 0:result.shape[1]]
    for pts, color in groups:
        for pt in pts:
            cx, cy = int(pt[0][0]), int(pt[0][1])
            circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= 25
            result[circle] = color
    return result


TRANSFORM_DISPLAY = {
    "gaussian_blur": ("Gaussian blur", transform_gaussian_blur),
    "mask": ("Mask", transform_mask),
    "roi_objects": ("ROI objects", transform_roi_objects),
    "analyze_object": ("Analyze object", transform_analyze_object),
    "pseudolandmarks": ("Pseudolandmarks", transform_pseudolandmarks),
}


def display_transformations(image_path: str):
    image, _, _ = pcv.readimage(filename=image_path)
    if image is None:
        return
    pcv.params.debug = None
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.canvas.manager.set_window_title("Transformation.py")
    rgb = image[:, :, ::-1]
    transforms_to_show = [
        ("Original", rgb),
        ("Gaussian blur", transform_gaussian_blur(image)[:, :, ::-1]),
        ("Mask", transform_mask(image)),
        ("ROI objects", transform_roi_objects(image)[:, :, ::-1]),
        ("Analyze object", transform_analyze_object(image)[:, :, ::-1]),
        ("Pseudolandmarks", transform_pseudolandmarks(image)[:, :, ::-1]),
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
    image, _, _ = pcv.readimage(filename=image_path)
    if image is None:
        return
    pcv.params.debug = None
    base = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    _, func = TRANSFORM_DISPLAY[transform_name]
    result = func(image)
    out_path = os.path.join(dst_dir, f"{base}_{transform_name}{ext}")
    pcv.print_image(img=result, filename=out_path)
    return out_path


def process_directory(src_dir: str, dst_dir: str, transform_name: str):
    """Apply a transform to all images in a directory and save results.

    Collects image paths by joining src_dir with each glob extension pattern
    (e.g. "*.JPG"), expanding matches, and extending the images list across
    all supported extensions.
    """
    os.makedirs(dst_dir, exist_ok=True)
    images = glob(os.path.join(src_dir, "*.JPG"))
    if not images:
        return
    for img_path in images:
        out = save_transform(img_path, transform_name, dst_dir)
    print(f"Processed {len(images)} images -> {dst_dir}")


def process_base_directory(base_dir: str, dst_dir: str = "transformed_images"):
    """Apply all transforms to all images in all subdirectories.

    Creates a 'transformed_images' sibling folder preserving
    the full subdirectory structure of base_dir.
    """
    base_dir = os.path.abspath(base_dir)
    dst_base = os.path.join(os.path.dirname(base_dir), dst_dir)
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
                             "subdirectories and save to transformed_images")
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
