"""
a. Detect all coins in the image (2 Marks)
    1) Use edge detection, to detect all coins in the image.
    2) Visualize the detected coins by outlining them in the image.

b. Segmentation of Each Coin (3 Marks)
    1) Apply region-based segmentation techniques to isolate individual coins from the
image.
    2) Provide segmented outputs for each detected coin.

c. Count the Total Number of Coins (2 Marks)
    1) Write a function to count the total number of coins detected in the image.
    2) Display the final count as an output
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_and_outline_coins(image_path, min_circularity=0.7, min_area_percentage=0.001):
    """
    Detects coins in an image using edge detection and circularity filtering.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at path: " + image_path)
    
    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Canny edge detection 
    edged = cv2.Canny(blurred, 30, 150)
    
    # Find contours from the edges
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the minimum area based on image dimensions
    image_area = image.shape[0] * image.shape[1]
    min_area_threshold = image_area * min_area_percentage
    
    # Filter contours based on circularity and scaled area
    coin_contours = []
    for contour in contours:
        # Calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Avoid division by zero
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Filter based on circularity and scaled minimum area
        if circularity >= min_circularity and area > min_area_threshold:
            coin_contours.append(contour)
    
    # Draw coin contours on a copy of the original image
    outlined = image.copy()
    cv2.drawContours(outlined, coin_contours, -1, (0, 255, 0), 2)
    
    print(f"Total contours detected: {len(contours)}")
    print(f"Coin contours after circle filtering: {len(coin_contours)}")
    
    return outlined, coin_contours, image, gray

def segment_coins(image, gray):
    """
    Uses a watershed-based segmentation to separate overlapping coins.
    """
    # Apply Otsu's thresholding after inverting the image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove noise with morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Dilate to obtain sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Use distance transform to find sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    
    # Determine unknown region by subtracting foreground from background
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling for the foreground objects
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that background becomes 1 instead of 0
    markers = markers + 1
    
    # Unknown region marked as 0
    markers[unknown == 255] = 0
    
    # Apply watershed algorithm; boundaries will be marked with -1
    image_watershed = image.copy()
    cv2.watershed(image_watershed, markers)
    image_watershed[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    
    # Extract individual coin segments based on unique markers
    coin_segments = {}
    for marker in np.unique(markers):
        if marker <= 1:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker] = 255
        coin = cv2.bitwise_and(image, image, mask=mask)
        coin_segments[marker] = coin
    
    return image_watershed, coin_segments

def count_coins(coin_contours):
    return len(coin_contours)

def main():
    # Change this path to the location of your coin image
    image_path = "images_coin/coins.png"
    
    # Detect coins and outline them on the image
    outlined, coin_contours, original, gray = detect_and_outline_coins(image_path)
    coin_count = count_coins(coin_contours)
    print("Total number of coins detected:", coin_count)
    
    # Segment coins
    watershed_img, _ = segment_coins(original, gray)
    
    # Create side-by-side display
    combined = np.hstack((original, outlined))
    
    # Create and display single window with both images
    cv2.namedWindow("Original | Coins Detected", cv2.WINDOW_NORMAL)
    cv2.imshow("Original | Coins Detected", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()