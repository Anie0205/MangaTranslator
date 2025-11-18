import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def detect_bubbles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 600:  # remove noise
            boxes.append([x, y, x + w, y + h])

    return boxes


def merge_boxes(boxes):
    if not boxes:
        return []

    arr = np.array(boxes)
    db = DBSCAN(eps=35, min_samples=1).fit(arr)

    merged = []
    for label in set(db.labels_):
        group = arr[db.labels_ == label]
        x1 = np.min(group[:, 0])
        y1 = np.min(group[:, 1])
        x2 = np.max(group[:, 2])
        y2 = np.max(group[:, 3])
        merged.append([x1, y1, x2, y2])

    return merged
