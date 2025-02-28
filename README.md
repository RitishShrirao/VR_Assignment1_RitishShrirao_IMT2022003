# Coin Detection and Image Stitching Project

## Table of Contents
1. [Introduction](#introduction)
2. [Installation and Dependencies](#installation-and-dependencies)
3. [Running the Code](#running-the-code)
4. [Methods Used](#methods-used)
5. [Results and Observations](#results-and-observations)
6. [Output Images](#output-images)
7. [Repository Structure](#repository-structure)

## Introduction
This project involves two main tasks:
1. **Coin Detection and Segmentation**: Detecting, outlining, and counting coins in an image using edge detection and segmentation techniques.
2. **Image Stitching**: Aligning and stitching multiple overlapping images to create a panorama using feature detection and transformation techniques.

## Installation and Dependencies
Ensure you have Python installed along with the following dependencies:
```sh
pip install opencv-python numpy matplotlib argparse glob2
```

## Running the Code
### Coin Detection
To detect and segment coins from an image, run:
```sh
python coin_detection.py --input_path <path_to_coin_image> --output_path <output_directory>
```
Example:
```sh
python coin_detection.py --input_path images/coins.jpg --output_path results/
```

### Image Stitching
To create a panorama from multiple images, run:
```sh
python image_stitching.py --input_path <folder_with_images> --output_path <output_directory>
```
Example:
```sh
python image_stitching.py --input_path images/panorama/ --output_path results/
```

## Methods Used
### Coin Detection and Segmentation
1. **Edge Detection**: Hough Circle Transform is used to detect circular objects (coins) in the image.
2. **Segmentation**: A mask is applied to segment individual coins from the background.
3. **Counting**: The total number of detected coins is displayed.

### Image Stitching
1. **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF) is used to detect key points.
2. **Feature Matching**: Key points are matched between overlapping images.
3. **Homography Estimation**: The transformation between images is computed.
4. **Panorama Generation**: Images are aligned and blended into a single panoramic image.

## Results and Observations
- The **coin detection** method effectively detects and outlines coins.
- The **segmentation method** isolates each coin and correctly counts them.
- The **stitching algorithm** successfully creates a panorama if images have sufficient overlap and distinct features.

## Output Images
Output images are saved in the specified output directory and include:
- `coins_detected.jpg`: Coins outlined in the image.
- `coins_segmented.jpg`: Segmented coins with labels.
- `coin_X.jpg`: Individual cropped images of each coin.
- `keypoints_combined.jpg`: Visualization of detected keypoints for image stitching.
- `panorama_result.jpg`: The final stitched panoramic image.

## Repository Structure
```
project_root/
├── coin_detection.py
├── images_coin
│   ├── coins.png
│   └── results
│       ├── coin_1.jpg
│       ├── coin_2.jpg
│       ├── coin_3.jpg
│       ├── coin_4.jpg
│       ├── coin_5.jpg
│       ├── coin_6.jpg
│       ├── coin_7.jpg
│       ├── coin_8.jpg
│       ├── coins_detected.jpg
│       └── coins_segmented.jpg
├── images_pan
│   ├── imgs
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── 3.png
│   └── results
│       ├── keypoints_combined.jpg
│       └── panorama_result.jpg
└── panaroma_creation.py

```
