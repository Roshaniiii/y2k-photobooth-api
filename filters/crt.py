import cv2
import numpy as np

def apply_crt(img):
    h, w = img.shape[:2]
    K = np.array([
        [w, 0, w / 2],
        [0, h, h / 2],
        [0, 0,     1]
    ], dtype=np.float32)
    D = np.array([0.15, 0.05, 0, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)