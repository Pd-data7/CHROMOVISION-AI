import cv2
import numpy as np


# ==========================================
# CHROMOVISION AI
# IMAGE SEGMENTATION ENGINE
# ==========================================


def create_binary_mask(hsv_image, lower, upper):

    """
    Generate binary segmentation mask
    using HSV thresholding.
    """

    mask = cv2.inRange(
        hsv_image,
        lower,
        upper
    )

    return mask


def remove_segmentation_noise(
    mask,
    method="Opening",
    kernel_size=5
):

    """
    Remove noise using morphological operations.
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    if method == "Opening":

        result = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

    elif method == "Closing":

        result = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

    elif method == "Both":

        result = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        result = cv2.morphologyEx(
            result,
            cv2.MORPH_CLOSE,
            kernel
        )

    else:

        result = mask.copy()

    return result


def extract_segmented_object(image, mask):

    """
    Extract selected object from original image.
    """

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return segmented


def calculate_segmentation_statistics(mask):

    """
    Calculate foreground and background statistics.
    """

    total_pixels = mask.size

    foreground_pixels = cv2.countNonZero(mask)

    background_pixels = (
        total_pixels - foreground_pixels
    )

    foreground_percentage = (
        foreground_pixels / total_pixels
    ) * 100

    background_percentage = (
        background_pixels / total_pixels
    ) * 100

    return {

        "Total Pixels": total_pixels,

        "Foreground Pixels": foreground_pixels,

        "Background Pixels": background_pixels,

        "Foreground Percentage": round(
            foreground_percentage, 2
        ),

        "Background Percentage": round(
            background_percentage, 2
        )

    }


def compare_segmentation(original, mask):

    """
    Generate original and segmented comparison.
    """

    segmented = extract_segmented_object(
        original,
        mask
    )

    return segmented