
import cv2
import numpy as np
import time

def apply_denoising(image, method="Median", kernel_size=5):
    """
    Apply noise removal to an image.

    Methods:
    1. Gaussian Blur
    2. Median Blur
    3. Bilateral Filter
    """

    start_time = time.perf_counter()

    if method == "Gaussian":
        processed = cv2.GaussianBlur(
            image, (kernel_size, kernel_size), 0
        )

    elif method == "Median":
        processed = cv2.medianBlur(
            image, kernel_size
        )

    elif method == "Bilateral":
        processed = cv2.bilateralFilter(
            image, 9, 75, 75
        )

    else:
        processed = image.copy()

    processing_time = (
        time.perf_counter() - start_time
    ) * 1000

    return processed, processing_time

def calculate_noise_ratio(mask, min_area=100):
    """
    Estimate the proportion of foreground pixels
    belonging to small isolated components.

    Lower ratio generally indicates fewer
    small isolated detections.
    """

    total_pixels = cv2.countNonZero(mask)

    if total_pixels == 0:
        return 0.0

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    small_component_pixels = 0

    for i in range(1, count):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            small_component_pixels += area

    noise_ratio = (
        small_component_pixels / total_pixels
    ) * 100

    return round(noise_ratio, 2)

def measure_processing_speed(processing_time_ms):
    """
    Calculate processing throughput in FPS.
    """

    if processing_time_ms <= 0:
        return 0

    fps = 1000 / processing_time_ms

    return round(fps, 2)