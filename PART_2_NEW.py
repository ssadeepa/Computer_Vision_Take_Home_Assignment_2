
# Name: Sadeepa P.M.A.S
# RegNo: EG/2019/3726
# Take Home Assignment 2

# Importing required libraries
import cv2
import numpy as np

def region_growing(image, seed, threshold):
    # Create a mask to store the segmented region
    region = np.zeros_like(image, dtype=np.uint8)
    region_points = [seed]
    while region_points:
        x, y = region_points.pop(0)
        if region[x, y] == 0 and abs(int(image[x, y]) - int(image[seed])) < threshold:
            region[x, y] = 255
            region_points.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
    return region

# Loading the image
image = cv2.imread('sample_img.jpg')

# Converting to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Defining seed point and threshold for region growing
seed_point = (50, 50)
threshold_value = 30
# Apply region growing segmentation
segmented_region = region_growing(gray_image, seed_point, threshold_value)



# Display original and segmented images
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Segmented Region', cv2.WINDOW_NORMAL)

cv2.imshow('Original Image', image)
cv2.imshow('Segmented Region', segmented_region)
cv2.waitKey(0)
cv2.destroyAllWindows()