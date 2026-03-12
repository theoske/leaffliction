# Transformation.py

Leaf image analysis program that extracts visual characteristics using 6 image transformations. Built with **OpenCV** and **PlantCV** for plant phenotyping workflows.

## Transformations

| # | Name | Description |
|---|------|-------------|
| 1 | **Gaussian blur** | Applies a 15x15 Gaussian kernel to smooth the image, reducing noise while preserving general structure |
| 2 | **Mask** | Isolates the leaf from the background using LAB color space thresholding (channel "a"), followed by fill, closing, erode, and dilate to clean up the binary mask |
| 3 | **ROI objects** | Highlights the leaf region of interest in green over a grayscale background, with a blue bounding rectangle around the largest detected contour |
| 4 | **Analyze object** | Draws the leaf contour in magenta, fits an ellipse, and computes principal axes (via PCA on contour points) shown as colored lines from the centroid |
| 5 | **Pseudolandmarks** | Places colored landmark points along the leaf contour, split into 3 groups (blue, orange, magenta) representing different contour sections |
| 6 | **Color histogram** | Plots pixel intensity distributions across 9 channels: blue, green, red (BGR), hue, saturation, value (HSV), lightness, green-magenta, blue-yellow (LAB). The histogram is computed only on masked (leaf) pixels |

## How the mask works

The mask is the foundation for most other transforms. The pipeline:

1. Convert BGR image to LAB color space, extract the "a" channel (green-magenta axis)
2. Binary threshold at 115 with `object_type="dark"` to select the green leaf
3. Fill small holes (< 300 pixels)
4. Morphological closing to connect nearby regions
5. Erode then dilate (3x3 kernel) to smooth edges

## Usage

### Display all transforms for a single image

```bash
python Transformation.py ./Apple/apple_healthy/image\ \(1\).JPG
```

Opens two matplotlib windows:
- A 2x3 grid showing: Original, Gaussian blur, Mask, ROI objects, Analyze object, Pseudolandmarks
- A separate color histogram plot

### Batch mode: process a directory

```bash
python Transformation.py -src Apple/apple_healthy/ -dst dst_directory -mask
```

Applies the mask transform to every image in the source directory and saves results in `dst_directory/`.

### Apply specific transforms in batch

Each transform can be selected with a flag:

```bash
python Transformation.py -src src_dir/ -dst dst_dir/ --gaussian-blur
python Transformation.py -src src_dir/ -dst dst_dir/ --mask
python Transformation.py -src src_dir/ -dst dst_dir/ --roi-objects
python Transformation.py -src src_dir/ -dst dst_dir/ --analyze-object
python Transformation.py -src src_dir/ -dst dst_dir/ --pseudolandmarks
python Transformation.py -src src_dir/ -dst dst_dir/ --color-histogram
```

Multiple flags can be combined. If no flag is given, **all transforms** are applied.

### Output file naming

Files are saved as `{original_name}_{transform_name}.{ext}`, for example:

```
image (1)_gaussian_blur.JPG
image (1)_mask.JPG
image (1)_roi_objects.JPG
image (1)_analyze_object.JPG
image (1)_pseudolandmarks.JPG
image (1)_color_histogram.png
```

Note: color histograms are always saved as `.png` since they are matplotlib plots.

## Code structure

| Function | Purpose |
|----------|---------|
| `create_mask(image)` | Core function: creates a binary leaf mask using PlantCV's LAB thresholding + morphological cleanup |
| `transform_gaussian_blur(image)` | Applies PlantCV Gaussian blur (15x15 kernel) |
| `transform_mask(image)` | Returns the binary mask |
| `transform_roi_objects(image)` | Green overlay on leaf + blue bounding box on grayscale background |
| `transform_analyze_object(image)` | Contour + fitted ellipse + PCA principal axes |
| `transform_pseudolandmarks(image)` | Evenly-spaced colored dots along contour (3 groups) |
| `transform_color_histogram(image)` | Multi-channel histogram (9 channels across BGR/HSV/LAB) |
| `display_transformations(path)` | Single image mode: shows all transforms in matplotlib |
| `save_transform(path, name, dst)` | Saves one transform result to disk |
| `process_directory(src, dst, name)` | Batch mode: applies one transform to all images in a directory |

## Dependencies

- **OpenCV** (`cv2`): image I/O, contour analysis, morphology, color conversions
- **PlantCV** (`plantcv`): LAB thresholding, binary fill, morphological ops, Gaussian blur
- **NumPy**: array operations, PCA computation
- **Matplotlib**: display and histogram plotting
