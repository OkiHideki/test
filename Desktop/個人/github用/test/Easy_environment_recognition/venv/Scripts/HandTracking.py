import numpy as np
import cv2
from hand_tracking_master.hand_tracker import HandTracker

def Hand_Tracking(img):
    PALM_MODEL_PATH = "./hand_tracking_master/palm_detection_without_custom_op.tflite"
    LANDMARK_MODEL_PATH = "./hand_tracking_master/hand_landmark.tflite"
    ANCHORS_PATH = "./hand_tracking_master/anchors.csv"

    CONNECTION_COLOR = (255, 0, 0)
    THICKNESS = 2

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
    ]

    detector = HandTracker(
        PALM_MODEL_PATH,
        LANDMARK_MODEL_PATH,
        ANCHORS_PATH,
        box_shift=0.2,
        box_enlarge=1.3
    )

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)
    points_index = 0
    if points is not None:
        oya = points[4]
        hito = points[8]
        dst = np.linalg.norm(oya - hito)
        for point in points:
            POINT_COLOR = (0, 255, 0)
            if points_index == 4 or points_index == 8:
                POINT_COLOR = (0, 0, 255)
            x, y = point
            cv2.circle(img, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            points_index += 1
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
    else:
        dst = None

    return img, dst