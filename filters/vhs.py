import cv2
import numpy as np

def apply_vhs(img):
    overlay = img.copy()
    for y in range(0, img.shape[0], 4):
        overlay[y:y+2] = (overlay[y:y+2] * 0.75).astype(np.uint8)

    b, g, r = cv2.split(img)
    r = np.roll(r, 3, axis=1)
    b = np.roll(b, -3, axis=1)
    colour_bleed = cv2.merge([b, g, r])

    return cv2.addWeighted(colour_bleed, 0.6, overlay, 0.4, 0)