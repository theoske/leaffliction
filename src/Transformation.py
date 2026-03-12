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
    "color_histogram",
]


def create_mask(image: np.ndarray) -> np.ndarray:
    """Create a binary mask to isolate the leaf from the background."""
    gray = pcv.rgb2gray_lab(rgb_img=image, channel="a")
    mask = pcv.threshold.binary(
        gray_img=gray, threshold=115, object_type="dark"
    )
    mask = pcv.fill(bin_img=mask, size=300)
    mask = pcv.closing(gray_img=mask)
    mask = pcv.erode(gray_img=mask, ksize=3, i=1)
    mask = pcv.dilate(gray_img=mask, ksize=3, i=1)
    return mask


def transform_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to the image."""
    return pcv.gaussian_blur(img=image, ksize=(15, 15), sigma_x=0)


def transform_mask(image: np.ndarray) -> np.ndarray:
    """Generate a binary mask of the leaf."""
    return create_mask(image)


def transform_roi_objects(image: np.ndarray) -> np.ndarray:
    """Show leaf region of interest: green overlay on leaf, blue bounding box."""
    mask = create_mask(image)
    h, w = image.shape[:2]

    gray_bg = cv2.cvtColor(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
    )
    green_overlay = gray_bg.copy()
    green_overlay[mask > 0] = (0, 255, 0)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(green_overlay, (x, y), (x + bw, y + bh), (255, 0, 0), 3)

    return green_overlay


def transform_analyze_object(image: np.ndarray) -> np.ndarray:
    """Analyze leaf shape: draw contour and principal axes."""
    mask = create_mask(image)
    result = image.copy()

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return result

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(result, [largest], -1, (255, 0, 255), 2)

    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        cv2.ellipse(result, ellipse, (255, 0, 0), 2)

    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        pts = largest.reshape(-1, 2).astype(np.float64)
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        for i, (val, vec) in enumerate(
            zip(eigenvalues[::-1], eigenvectors.T[::-1])
        ):
            length = int(np.sqrt(val) * 3)
            endpoint = (int(cx + vec[0] * length), int(cy + vec[1] * length))
            color = (255, 0, 0) if i == 0 else (0, 0, 255)
            cv2.line(result, (cx, cy), endpoint, color, 2)

    return result


def transform_pseudolandmarks(image: np.ndarray) -> np.ndarray:
    """Place pseudolandmarks along the leaf contour in 3 groups."""
    mask = create_mask(image)
    result = image.copy()

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return result

    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2)
    n = len(pts)
    third = n // 3

    groups = [
        (pts[:third], (255, 0, 0)),
        (pts[third: 2 * third], (0, 128, 255)),
        (pts[2 * third:], (255, 0, 255)),
    ]

    for group_pts, color in groups:
        step = max(1, len(group_pts) // 20)
        for pt in group_pts[::step]:
            cv2.circle(result, tuple(pt), 5, color, -1)

    return result


def transform_color_histogram(image: np.ndarray, mask: np.ndarray = None):
    """
    Generate a multi-channel color histogram figure.
    Returns a matplotlib figure (not an image array).
    """
    if mask is None:
        mask = create_mask(image)

    channels = {
        "blue": (None, 0),
        "green": (None, 1),
        "red": (None, 2),
        "hue": (cv2.COLOR_BGR2HSV, 0),
        "saturation": (cv2.COLOR_BGR2HSV, 1),
        "value": (cv2.COLOR_BGR2HSV, 2),
        "lightness": (cv2.COLOR_BGR2Lab, 0),
        "green-magenta": (cv2.COLOR_BGR2Lab, 1),
        "blue-yellow": (cv2.COLOR_BGR2Lab, 2),
    }

    colors = {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "hue": "purple",
        "saturation": "cyan",
        "value": "orange",
        "lightness": "gray",
        "green-magenta": "magenta",
        "blue-yellow": "yellow",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    converted_cache = {}

    for name, (conversion, ch_idx) in channels.items():
        if conversion not in converted_cache:
            if conversion is None:
                converted_cache[conversion] = image.copy()
            else:
                converted_cache[conversion] = cv2.cvtColor(image, conversion)
        converted = converted_cache[conversion]

        hist = cv2.calcHist(
            [converted], [ch_idx], mask, [256], [0, 256]
        )
        total = hist.sum()
        if total > 0:
            hist = (hist / total) * 100

        ax.plot(hist, color=colors[name], label=name, linewidth=1)

    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.legend(title="color Channel", loc="upper right")
    ax.set_xlim([0, 256])

    return fig


TRANSFORM_DISPLAY = {
    "gaussian_blur": ("Gaussian blur", transform_gaussian_blur),
    "mask": ("Mask", transform_mask),
    "roi_objects": ("ROI objects", transform_roi_objects),
    "analyze_object": ("Analyze object", transform_analyze_object),
    "pseudolandmarks": ("Pseudolandmarks", transform_pseudolandmarks),
    "color_histogram": ("Color histogram", None),
}


def display_transformations(image_path: str):
    """Display all transformations for a single image using matplotlib."""
    image = cv2.imread(image_path)
    if image is None:
        #print(f"Error: cannot read {image_path}")
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

    hist_fig = transform_color_histogram(image)
    hist_fig.canvas.manager.set_window_title("Color histogram")

    plt.show()


def save_transform(image_path: str, transform_name: str, dst_dir: str):
    """Apply one transform to an image and save the result."""
    image = cv2.imread(image_path)
    if image is None:
        #print(f"Error: cannot read {image_path}")
        return

    pcv.params.debug = None

    base = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]

    if transform_name == "color_histogram":
        mask = create_mask(image)
        fig = transform_color_histogram(image, mask)
        out_path = os.path.join(dst_dir, f"{base}_{transform_name}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        _, func = TRANSFORM_DISPLAY[transform_name]
        result = func(image)
        out_path = os.path.join(dst_dir, f"{base}_{transform_name}{ext}")
        cv2.imwrite(out_path, result)

    return out_path


def process_directory(src_dir: str, dst_dir: str, transform_name: str):
    """Apply a transform to all images in a directory and save results."""
    os.makedirs(dst_dir, exist_ok=True)

    extensions = ("*.JPG", "*.jpg", "*.png", "*.PNG", "*.jpeg", "*.JPEG")
    images = []
    for ext in extensions:
        images.extend(glob(os.path.join(src_dir, ext)))

    if not images:
        #print(f"No images found in {src_dir}")
        return

    for img_path in images:
        out = save_transform(img_path, transform_name, dst_dir)
        #print(f"  {out}")

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
            #print(f"[{rel_path}] [{tf}]")
            process_directory(dirpath, dst_dir, tf)


def main():
    parser = argparse.ArgumentParser(
        description="Leaf image transformation: Gaussian blur, mask, "
                    "ROI objects, analyze object, pseudolandmarks, "
                    "color histogram"
    )
    parser.add_argument("image", nargs="?",
                        help="Path to a single image (displays all transforms)")
    parser.add_argument("-src", "--source",
                        help="Source directory of images (batch mode)")
    parser.add_argument("-dst", "--destination",
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
            #print(f"[{tf}]")
            process_directory(args.source, args.destination, tf)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
