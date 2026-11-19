import cv2
import numpy as np


# ==========================================
# CHROMOVISION AI
# ADVANCED 10-COLOUR DETECTION ENGINE
# ==========================================


# HSV colour ranges
# OpenCV HSV: H = 0-179, S = 0-255, V = 0-255

COLOR_RANGES = {

    "Red": [
        (
            np.array([0, 100, 100]),
            np.array([10, 255, 255])
        ),
        (
            np.array([170, 100, 100]),
            np.array([179, 255, 255])
        )
    ],

    "Orange": [
        (
            np.array([11, 100, 100]),
            np.array([20, 255, 255])
        )
    ],

    "Yellow": [
        (
            np.array([21, 100, 100]),
            np.array([35, 255, 255])
        )
    ],

    "Green": [
        (
            np.array([36, 80, 50]),
            np.array([85, 255, 255])
        )
    ],

    "Cyan": [
        (
            np.array([86, 80, 50]),
            np.array([99, 255, 255])
        )
    ],

    "Blue": [
        (
            np.array([100, 80, 50]),
            np.array([130, 255, 255])
        )
    ],

    "Purple": [
        (
            np.array([131, 80, 50]),
            np.array([149, 255, 255])
        )
    ],

    "Pink": [
        (
            np.array([150, 50, 100]),
            np.array([169, 255, 255])
        )
    ],

    # White: low saturation and high brightness
    "White": [
        (
            np.array([0, 0, 190]),
            np.array([179, 45, 255])
        )
    ],

    # Black: very low brightness
    "Black": [
        (
            np.array([0, 0, 0]),
            np.array([179, 255, 45])
        )
    ]
}


# Bounding box colours in BGR format

COLOR_BGR = {

    "Red": (0, 0, 255),

    "Orange": (0, 140, 255),

    "Yellow": (0, 255, 255),

    "Green": (0, 255, 0),

    "Cyan": (255, 255, 0),

    "Blue": (255, 0, 0),

    "Purple": (180, 0, 180),

    "Pink": (203, 192, 255),

    "White": (255, 255, 255),

    "Black": (0, 0, 0)
}


def detect_colors(image, min_area=500, custom_ranges=None):

    """
    CHROMOVISION AI
    Advanced Multi-Colour Object Detection Engine.

    Parameters:
        image: Input BGR image.
        min_area: Minimum contour area.
        custom_ranges: Optional custom HSV colour ranges.

    Returns:
        output: Image with bounding boxes.
        masks: Individual binary masks.
        objects: Detected object information.
    """

    # STEP 1: Validate input image

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid input image.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    if min_area < 0:
        raise ValueError("Minimum area cannot be negative.")

    # STEP 2: Select colour ranges

    active_ranges = (
        custom_ranges if custom_ranges is not None
        else COLOR_RANGES
    )

    # STEP 3: Convert BGR into HSV

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # STEP 4: Create output image

    output = image.copy()

    masks = {}
    objects = []

    # STEP 5: Morphological kernel

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    global_object_id = 0

    # STEP 6: Detect all selected colours

    for color_name, ranges in active_ranges.items():

        # Create empty mask

        mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8
        )

        # Apply HSV thresholding

        for lower, upper in ranges:

            lower = np.asarray(
                lower,
                dtype=np.uint8
            )

            upper = np.asarray(
                upper,
                dtype=np.uint8
            )

            current_mask = cv2.inRange(
                hsv,
                lower,
                upper
            )

            mask = cv2.bitwise_or(
                mask,
                current_mask
            )

        # STEP 7: Remove small noise

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # STEP 8: Fill small gaps

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        masks[color_name] = mask

        # STEP 9: Find contours

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        drawing_color = COLOR_BGR.get(
            color_name,
            (255, 255, 255)
        )

        # STEP 10: Analyse detected objects

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            global_object_id += 1

            # Bounding rectangle

            x, y, w, h = cv2.boundingRect(contour)

            # Object centre

            center_x = x + w // 2
            center_y = y + h // 2

            # Perimeter

            perimeter = cv2.arcLength(
                contour,
                True
            )

            # Bounding box

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                drawing_color,
                3
            )

            # Centre point

            cv2.circle(
                output,
                (center_x, center_y),
                5,
                drawing_color,
                -1
            )

            # Object label

            label = f"{color_name} ID:{global_object_id}"

            cv2.putText(
                output,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                drawing_color,
                2
            )

            # Store object information

            objects.append({

                "ID": global_object_id,

                "Color": color_name,

                "Area": int(area),

                "Perimeter": round(perimeter, 2),

                "X": int(center_x),

                "Y": int(center_y),

                "Width": int(w),

                "Height": int(h),

                "Aspect Ratio": round(w / h, 2) if h else 0

            })

    # STEP 11: Return results

    return output, masks, objects