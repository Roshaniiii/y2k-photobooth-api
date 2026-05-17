import numpy as np

def apply_y2k(img):
    img_float = img.astype('float32')
    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * 1.2, 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * 0.85, 0, 255)
    result = np.clip(img_float * 0.9 + 20, 0, 255)
    return result.astype('uint8')