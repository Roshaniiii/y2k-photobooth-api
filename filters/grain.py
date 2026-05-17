import numpy as np

def apply_grain(img):
    noise = np.random.normal(0, 25, img.shape).astype('int16')
    result = np.clip(img.astype('int16') + noise, 0, 255)
    return result.astype('uint8')