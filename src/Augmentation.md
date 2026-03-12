# Augmentation.py

Data augmentation program that applies 6 image transformations to balance an unbalanced dataset.

## Transformations

| # | Name | Description |
|---|------|-------------|
| 1 | **Flip** | Horizontal mirror of the image |
| 2 | **Rotate** | Random rotation between -30 and +30 degrees. The output canvas expands to fit the rotated image, with reflected borders to avoid black edges |
| 3 | **Skew** | Perspective (projective) transformation. Each corner is randomly shifted inward by up to 15% of image dimensions, creating a 3D viewing angle effect |
| 4 | **Shear** | Affine shear along the horizontal axis (factor between -0.3 and 0.3). Slants the image left or right while preserving parallel horizontal lines |
| 5 | **Crop** | Zooms into the center of the image (1.2x to 1.6x) and resizes back to original dimensions, simulating a scale change |
| 6 | **Distortion** | Barrel/pincushion lens distortion using OpenCV's camera model. Simulates the warping effect of a wide-angle lens |

## Usage

### Augment a single image

```bash
python Augmentation.py ./Apple/apple_healthy/image\ \(1\).JPG
```

This creates 6 new images **in the same directory** as the original:

```
image (1)_Flip.JPG
image (1)_Rotate.JPG
image (1)_Skew.JPG
image (1)_Shear.JPG
image (1)_Crop.JPG
image (1)_Distortion.JPG
```

### Augment to a specific output directory

```bash
python Augmentation.py ./Apple/apple_healthy/image\ \(1\).JPG -o ./output/
```

### Balance the entire dataset

```bash
python Augmentation.py --balance
```

This will:

1. Copy the full dataset from `IMAGES_PATH` (defined in `global_var.py`) into an `augmented_directory/` folder at the same level.
2. **Split 20% of each category's images** into an `evaluation_images/` folder (sibling of `augmented_directory/`). These images are kept raw and are not augmented.
3. Identify under-represented categories using `Distribution.get_distribution()`.
4. Generate augmented images for each under-represented category until all categories have the same number of images as the largest one.
5. Augmented files are named `originalname_aug{n}_{Transform}.JPG` and placed directly in the category folder.

You can also specify a custom output path:

```bash
python Augmentation.py --balance -o ./my_augmented_dataset/
```

## Code structure

| Function | Purpose |
|----------|---------|
| `apply_flip(image)` | Returns horizontally flipped image |
| `apply_rotate(image)` | Returns randomly rotated image |
| `apply_skew(image)` | Returns perspective-warped image |
| `apply_shear(image)` | Returns horizontally sheared image |
| `apply_crop(image)` | Returns center-cropped and resized image |
| `apply_distortion(image)` | Returns barrel-distorted image |
| `augment_image(path, output_dir)` | Applies all 6 transforms to one image and saves them |
| `select_categories_to_augment(data_path)` | Analyzes distribution and returns list of (name, current_size, target_size) for categories below the max |
| `split_evaluation(data_path, eval_ratio, seed)` | Moves `eval_ratio` (default 20%) of each category's images into `evaluation_images/` |
| `balance_dataset(data_path, output_path)` | Copies dataset to `augmented_directory/`, splits evaluation set, then augments to balance all categories |

## Dependencies

- **OpenCV** (`cv2`): all image I/O and transformations
- **NumPy**: array operations and random parameter generation
- **Distribution.py**: `get_distribution()` to analyze category sizes
- **global_var.py**: `IMAGES_PATH` constant pointing to the image dataset
