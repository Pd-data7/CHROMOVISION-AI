import cv2
import numpy as np
import pandas as pd
import time


# ==========================================
# CHROMOVISION AI
# ADVANCED ANALYTICS ENGINE
# ==========================================


def analyze_detection_data(objects):
    """
    Generate statistical analysis
    from detected objects.
    """

    if not objects:
        return {
            "Total Objects": 0,
            "Color Distribution": {},
            "Average Area": 0,
            "Largest Object": 0
        }

    df = pd.DataFrame(objects)

    color_distribution = (
        df["Color"].value_counts().to_dict()
    )

    average_area = df["Area"].mean()

    largest_object = df["Area"].max()

    return {

        "Total Objects": len(objects),

        "Color Distribution": color_distribution,

        "Average Area": round(
            average_area, 2
        ),

        "Largest Object": round(
            largest_object, 2
        )

    }


# ==========================================
# PROCESSING SPEED
# ==========================================

def calculate_fps(processing_time_ms):

    if processing_time_ms <= 0:
        return 0

    return round(
        1000 / processing_time_ms, 2
    )


def benchmark_processing(image, detection_function,
                         iterations=10):

    """
    Measure average processing time
    and throughput.
    """

    timings = []

    for _ in range(iterations):

        start = time.perf_counter()

        detection_function(image)

        end = time.perf_counter()

        elapsed = (end - start) * 1000

        timings.append(elapsed)

    average_time = np.mean(timings)

    return {

        "Average Processing Time (ms)": round(
            average_time, 2
        ),

        "Minimum Processing Time (ms)": round(
            min(timings), 2
        ),

        "Maximum Processing Time (ms)": round(
            max(timings), 2
        ),

        "Processing FPS": calculate_fps(
            average_time
        )

    }


# ==========================================
# SEGMENTATION EVALUATION
# ==========================================

def calculate_segmentation_metrics(
    predicted_mask,
    ground_truth_mask
):

    """
    Evaluate predicted segmentation
    against manually labelled ground truth.
    """

    predicted = predicted_mask > 0

    ground_truth = ground_truth_mask > 0

    tp = np.logical_and(
        predicted, ground_truth
    ).sum()

    fp = np.logical_and(
        predicted, ~ground_truth
    ).sum()

    fn = np.logical_and(
        ~predicted, ground_truth
    ).sum()

    tn = np.logical_and(
        ~predicted, ~ground_truth
    ).sum()

    precision = tp / (tp + fp) if tp + fp else 0

    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall else 0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0
    )

    return {

        "Accuracy (%)": round(accuracy * 100, 2),

        "Precision (%)": round(precision * 100, 2),

        "Recall (%)": round(recall * 100, 2),

        "F1 Score (%)": round(f1 * 100, 2),

        "IoU (%)": round(iou * 100, 2)

    }


# ==========================================
# DETECTION HISTORY
# ==========================================

def save_detection_history(objects):

    """
    Convert detection results into
    a structured Pandas DataFrame.
    """

    if not objects:
        return pd.DataFrame()

    df = pd.DataFrame(objects)

    df["Timestamp"] = pd.Timestamp.now()

    return df


# ==========================================
# CSV EXPORT
# ==========================================

def export_detection_report(objects):

    """
    Generate downloadable CSV report.
    """

    df = save_detection_history(objects)

    if df.empty:
        return None

    return df.to_csv(index=False)


# ==========================================
# IMAGE NOISE ANALYSIS
# ==========================================

def calculate_mask_noise(mask, min_area=100):

    """
    Estimate percentage of foreground pixels
    belonging to small connected components.
    """

    total_pixels = cv2.countNonZero(mask)

    if total_pixels == 0:
        return 0.0

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    small_pixels = 0

    for i in range(1, count):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            small_pixels += area

    return round(
        (small_pixels / total_pixels) * 100,
        2
    )