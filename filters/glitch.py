import numpy as np

def apply_glitch(img):
    out = img.copy()
    h = img.shape[0]

    for _ in range(12):
        y = np.random.randint(0, h - 20)
        strip_height = np.random.randint(2, 20)
        shift = np.random.randint(-30, 30)
        out[y:y + strip_height] = np.roll(
            out[y:y + strip_height], shift, axis=1
        )
    return out