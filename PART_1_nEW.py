
# Name: Sadeepa P.M.A.S
# RegNo: EG/2019/3726
# Take Home Assignment 2


# Import libraries
import numpy as np
import cv2

def generateImage(width, height):
    # Creating an empty grayscale image
    image = np.zeros((height, width), dtype=np.uint8)

    # Setting pixel values for distinct image regions (Background: Black)
    image[:, :] = 0

    # Square (Gray)
    square_size = width // 2
    square_x = 0
    square_y = (height - square_size) // 2
    image[square_y:square_y+square_size, square_x:square_x+square_size] = 128  # color is gray

    # Circle (White)
    circle_radius = width // 5
    circle_center = (width - circle_radius, height // 2)
    cv2.circle(image, circle_center, circle_radius, 255, -1)  # color is white

    return image

def addGaussianNoise(image):
    # Converting image to floating-point data type
    image_float = image.astype(np.float32)
    # Generate Gaussian noise with specified mean and standard deviation
    # Passing mean and stddev
    noise = np.random.normal(0, 50, size=image.shape).astype(np.float32)
    # Adding noise to the image
    img_noised = image_float + noise
    # Ensure that the pixel values are limited to the acceptable range of [0, 255].
    noisy_img = np.clip(img_noised, 0, 255).astype(np.uint8)
    return noisy_img

# Displaying image
generatedImg = generateImage(300,300)
cv2.imshow("Image with 3 Pixel Values", generatedImg)

# Display the noisy image
noisyImage = addGaussianNoise(generatedImg)
cv2.imshow("Noise added Image", noisyImage)

# Applying Otsu's thresholding
_, otsuThreshold = cv2.threshold(noisyImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Displaying the thresholded image
cv2.imshow("Otsu's Thresholding", otsuThreshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
