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
import argparse

def detect_and_outline_coins(image_path, param1=50, param2=30, minRadius=20, maxRadius=100):
    """
    Detects coins in an image using Hough Circle Transform.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at path: " + image_path)
    
    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use Hough Circle Transform to detect circular objects (coins)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1,
        minDist=50,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    
    # Create contours for each detected circle
    coin_contours = []
    outlined = image.copy()
    
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.uint16(np.around(circles))
        
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            
            # Draw the circle on the outlined image
            cv2.circle(outlined, center, radius, (0, 255, 0), 2)
            
            # Create a contour for this circle for segmentation
            points = []
            for angle in range(0, 360, 5):  # Sample the circle at every 5 degrees
                x = int(center[0] + radius * np.cos(np.radians(angle)))
                y = int(center[1] + radius * np.sin(np.radians(angle)))
                points.append([x, y])
                
            contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
            coin_contours.append(contour)
    
    print(f"Total coins detected: {len(coin_contours)}")
    
    return outlined, coin_contours, image, gray

def segment_coins(image, contours):
    """
    Segments individual coins based on the detected contours.
    Returns individual coin images and a visualization of all segmented coins.
    """
    # Create a black mask for visualization
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    # Create a dictionary to store individual coin segments
    coin_segments = {}
    
    # Create a copy of the original image for visualization
    segmented_display = image.copy()
    
    # Process each contour
    for i, contour in enumerate(contours):
        # Create a mask for this specific coin
        coin_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(coin_mask, [contour], -1, 255, -1)
        
        # Apply the mask to extract the coin
        coin = cv2.bitwise_and(image, image, mask=coin_mask)
        
        # Store the segmented coin
        coin_segments[i] = coin
        
        # Add this coin to the visualization with a random color
        color = np.random.randint(0, 255, (3,)).tolist()
        cv2.drawContours(mask, [contour], -1, color, -1)
        
        # Draw contour number on the segmented display
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(segmented_display, f"#{i+1}", (cX-20, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Create a blended visualization
    alpha = 0.6
    segmented_viz = cv2.addWeighted(segmented_display, 1-alpha, mask, alpha, 0)
    
    return segmented_viz, coin_segments

def count_coins(coin_contours):
    """Count the total number of coins detected."""
    return len(coin_contours)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Detect and segment coins in an image.")
    parser.add_argument("--input_path", required=True, help="Path to the input image.")
    parser.add_argument("--output_path", required=True, help="Path to the output directory.")
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    
    # Detect coins and outline them on the image
    outlined, coin_contours, original, gray = detect_and_outline_coins(input_path)
    coin_count = count_coins(coin_contours)
    print("Total number of coins detected:", coin_count)
    
    # Segment coins using the improved method
    segmented_viz, coin_segments = segment_coins(original, coin_contours)
    
    # Save results
    cv2.imwrite(f"{output_path}/coins_detected.jpg", outlined)
    cv2.imwrite(f"{output_path}/coins_segmented.jpg", segmented_viz)
    
    # Optional: Save individual coin segments
    for idx, coin in coin_segments.items():
        # Create a bounding box for the coin to crop it more tightly
        y_indices, x_indices = np.where(np.any(coin > 0, axis=2))
        if len(y_indices) > 0 and len(x_indices) > 0:
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            # Add some padding
            pad = 5
            y_min = max(0, y_min - pad)
            y_max = min(coin.shape[0], y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(coin.shape[1], x_max + pad)
            
            cropped = coin[y_min:y_max, x_min:x_max]
            cv2.imwrite(f"{output_path}/coin_{idx+1}.jpg", cropped)

if __name__ == "__main__":
    main()
