import cv2
import numpy as np

def apply_chroma(img):
    b, g, r = cv2.split(img)
    r = np.roll(r, 5, axis=1)
    b = np.roll(b, -5, axis=1)
    g = np.roll(g, 2, axis=0)
    return cv2.merge([b, g, r])