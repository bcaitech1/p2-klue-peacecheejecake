import numpy as np
import cv2


##########################################
# PREPROCESSING ##########################
##########################################


def preprocess(image: np.ndarray, size=256, segment=True, blur=False):
    # Normalize
    image = cv2.normalize(image, 0, 255)

    # Segmentation
    

    # Remove noise - Guassian blur
    if blur:
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.bilateralFilter(image, 5, 75, 75)

    return image

