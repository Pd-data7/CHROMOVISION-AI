import cv2
import numpy as np


# ==========================================
# CHROMOVISION AI
# OBJECT GEOMETRY ANALYSIS
# ==========================================


def analyze_contours(mask, min_area=500):

    """
    Analyze object contours and calculate
    geometric properties.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for index, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Calculate perimeter
        perimeter = cv2.arcLength(
            contour,
            True
        )

        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio
        aspect_ratio = w / h if h != 0 else 0

        # Calculate centroid
        moments = cv2.moments(contour)

        if moments["m00"] != 0:

            cx = int(
                moments["m10"] / moments["m00"]
            )

            cy = int(
                moments["m01"] / moments["m00"]
            )

        else:
            cx, cy = x + w // 2, y + h // 2

        # Calculate circularity
        if perimeter > 0:

            circularity = (
                4 * np.pi * area
            ) / (perimeter ** 2)

        else:
            circularity = 0

        # Store measurements
        results.append({

            "Object ID": len(results) + 1,

            "Area": round(area, 2),

            "Perimeter": round(perimeter, 2),

            "Centroid X": cx,

            "Centroid Y": cy,

            "Width": w,

            "Height": h,

            "Aspect Ratio": round(
                aspect_ratio, 2
            ),

            "Circularity": round(
                circularity, 3
            )

        })

    return results


def draw_contour_analysis(image, mask, min_area=500):

    """
    Draw object boundaries and centroid positions.
    """

    output = image.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    object_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        object_count += 1

        x, y, w, h = cv2.boundingRect(contour)

        moments = cv2.moments(contour)

        if moments["m00"] != 0:

            cx = int(
                moments["m10"] / moments["m00"]
            )

            cy = int(
                moments["m01"] / moments["m00"]
            )

        else:
            cx = x + w // 2
            cy = y + h // 2

        # Draw contour
        cv2.drawContours(
            output,
            [contour],
            -1,
            (0, 255, 255),
            2
        )

        # Draw bounding rectangle
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Draw centroid
        cv2.circle(
            output,
            (cx, cy),
            5,
            (255, 0, 255),
            -1
        )

        # Add object label
        cv2.putText(
            output,
            f"Object {object_count}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return output, object_count