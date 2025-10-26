""""""

from typing import Optional

import numpy
import cv2
import scipy.ndimage as ndimage


def equalize(image_in: numpy.ndarray, mode: str, ksize:int) -> numpy.ndarray:
    if mode == "histogram":
        return cv2.equalizeHist(image_in)
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(ksize, ksize))
        return clahe.apply(image_in)
    return image_in


def blur(image_in: numpy.ndarray, mode: str, ksize: int) -> numpy.ndarray:
    if mode == "medianBlur":
        return cv2.medianBlur(image_in, ksize)
    if mode == "blur":
        return cv2.blur(image_in, (ksize, ksize))
    if mode == "GaussianBlur":
        return cv2.GaussianBlur(image_in, (ksize, ksize), 0)
    if mode == "bilateralFilter":
        return cv2.bilateralFilter(image_in, ksize, sigmaColor=75, sigmaSpace=75)
    return image_in


def binarize(image_in: numpy.ndarray, mode: str, ksize: int = None, thresh: int = 127) -> numpy.ndarray:
    if mode == "thresh_otsu":
        thresh, image_out = cv2.threshold(
            src=image_in,
            thresh=thresh,
            maxval=255,
            type=cv2.THRESH_OTSU,
        )
        return image_out
    if mode == "thresh_binary":
        thresh, image_out = cv2.threshold(
            src=image_in,
            thresh=thresh,
            maxval=255,
            type=cv2.THRESH_BINARY,
        )
        return image_out
    if mode == "adaptive_thresh_gaussian":
        return cv2.adaptiveThreshold(
            src=image_in,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=ksize,
            C=2,
        )
    if mode == "adaptive_thresh_mean":
        return cv2.adaptiveThreshold(
            src=image_in,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=ksize,
            C=2,
        )
    return image_in


def skeletonize(
    image_in: numpy.ndarray,
    skeleton_radius: int = 9,
    erosion_size: int = 3
):
    ##### distance transform and skeleton
    distance_img = cv2.distanceTransform(
        src=image_in,
        distanceType=[cv2.DIST_L1, cv2.DIST_L2, cv2.DIST_C][2],
        maskSize=[3, 5][1],
    )
    distance_img = cv2.normalize(distance_img, None, 0, 1.0, cv2.NORM_MINMAX)

    # get the skeleton    
    laplace_img = ndimage.morphological_laplace(distance_img, (skeleton_radius, skeleton_radius))
    laplace_img = cv2.normalize(laplace_img, None, 0, 1.0, cv2.NORM_MINMAX)
    skeleton_mask = laplace_img < laplace_img.max() / 7
    skeleton_image = image_in.copy()
    skeleton_image *= 0
    skeleton_image[skeleton_mask] = 255.0
      
    # erode to thin the pipe
    if erosion_size is not None:
        erosion_kernel = numpy.ones((erosion_size, erosion_size), numpy.uint8)
        skeleton_image_eroded = cv2.erode(skeleton_image, erosion_kernel)
    else:
        skeleton_image_eroded = skeleton_image.copy()

    return distance_img, laplace_img, skeleton_mask, skeleton_image, skeleton_image_eroded
