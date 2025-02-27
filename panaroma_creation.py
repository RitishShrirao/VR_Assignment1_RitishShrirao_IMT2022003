"""
a. Extract Key Points (1 Mark)
    1) Detect key points in overlapping images.
b. Image Stitching (2 Marks)
    1) Use the extracted key points to align and stitch the images into a single panorama.
    2) Provide the final panorama image as output.
"""

import cv2
import numpy as np
import glob

def load_images(image_folder):
    """
    Load images of common formats
    """
    # Create patterns for common image formats
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    image_paths = []
    
    # Gather all images with supported extensions
    for extension in image_extensions:
        image_paths.extend(glob.glob(f"{image_folder}/{extension}"))
    
    image_paths = sorted(image_paths)
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            print(f"Loaded image: {path}")
        else:
            print(f"Warning: Unable to load image at {path}")
    
    return images

def display_keypoints(images):
    """
    Detects and displays keypoints for all images side by side.
    """
    orb = cv2.ORB_create(nfeatures=2000)
    keypoint_images = []
    
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        print(f"Image {idx}: {len(kp)} keypoints detected")
        
        # Draw keypoints on the image
        img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

        # Add a label to the image
        cv2.putText(img_kp, f"Image {idx}: {len(kp)} keypoints", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        keypoint_images.append(img_kp)
    
    # Resize images to the same height
    max_height = max(img.shape[0] for img in keypoint_images)
    resized_images = []
    
    for img in keypoint_images:
        if img.shape[0] != max_height:
            aspect_ratio = img.shape[1] / img.shape[0]
            new_width = int(max_height * aspect_ratio)
            resized_img = cv2.resize(img, (new_width, max_height))
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    
    # Concatenate images horizontally
    if resized_images:
        combined_image = np.hstack(resized_images)
        cv2.imshow("All Keypoints", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("keypoints_combined.jpg", combined_image)
        print("Combined keypoints image saved as 'keypoints_combined.jpg'")

def stitch_images(images):
    """
    Stitches multiple images into a panorama using OpenCV's built-in Stitcher (SCANS mode).
    """
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, panorama = stitcher.stitch(images)
    return panorama

def main():
    # Change this path to your folder containing overlapping images for panorama
    image_folder = "images_pan/imgs"
    
    images = load_images(image_folder)
    if len(images) < 2:
        print("Need at least two overlapping images to create a panorama.")
        return
    
    print(f"Loaded {len(images)} images")
    
    display_keypoints(images)
    
    # Try different image resizing to help with stitching
    resized_images = []
    for img in images:
        # Resize images if they are very large
        width = int(img.shape[1] * 0.75)
        height = int(img.shape[0] * 0.75)
        resized_img = cv2.resize(img, (width, height))
        resized_images.append(resized_img)
    
    print("Attempting to stitch images...")
    panorama = stitch_images(images)
    
    # If original stitching fails, try with resized images
    if panorama is None:
        print("Trying again with resized images...")
        panorama = stitch_images(resized_images)
    
    if panorama is not None:
        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("panorama_result.jpg", panorama)
        print("Panorama created and saved as 'panorama_result.jpg'.")
    else:
        print("Panorama stitching failed. Make sure your images have enough overlap and distinct features.")

if __name__ == "__main__":
    main()