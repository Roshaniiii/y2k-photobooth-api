import os
import cv2
import numpy as np
from typing import Optional, Tuple, List

# ── Cascade ───────────────────────────────────────────────────────────────────
_FACE_XML: str = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"

# ── Asset path ────────────────────────────────────────────────────────────────
_ASSET_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "star.png"
)

# ── Star PNG intrinsics (397×395, star centre at 198,200) ─────────────────────
_SRC_W:        int = 397
_SRC_H:        int = 395
_SRC_ANCHOR_X: int = 198
_SRC_ANCHOR_Y: int = 200

# ── Recolor targets (BGR) — applied to the star mask ─────────────────────────
# Reference image shows: blue, green, orange, purple, pink, red stars
_STAR_COLOURS: List[Tuple[int, int, int]] = [
    (219, 107,  73),   # Blue      #4B6BDB
    ( 60, 160,  60),   # Green     #3CA03C
    ( 30, 140, 220),   # Orange    #DC8C1E
    (180,  80, 140),   # Purple    #8C50B4
    (140, 105, 220),   # Pink      #DC69 8C
    ( 40,  40, 180),   # Red       #B42828
    (200, 200,  60),   # Teal      #3CC8C8
]

# ── Star placement config ─────────────────────────────────────────────────────
# Fixed positions as fractions of (face_w, face_h)
# Rules:
#   • fy > 0.20  → below forehead hairline
#   • fy < 0.82  → above chin
#   • fx 0.12-0.88 → inside face width
#   • Avoid eye zone: fy 0.28-0.48 with fx 0.25-0.75 (nose bridge + eye area)
# (pfx, pfy, size_frac, colour_idx, rotation_deg)
_STAR_PLACEMENTS: List[Tuple[float, float, float, int, float]] = [
    (0.18, 0.22, 0.15, 0,  15),   # forehead left
    (0.60, 0.20, 0.1, 1, -12),   # forehead right
    (0.12, 0.56, 0.13, 2,   8),   # left cheek mid
    (0.88, 0.50, 0.15, 3, -10),   # right cheek mid
    (0.20, 0.70, 0.09, 4,  20),   # left cheek lower
    (0.78, 0.68, 0.09, 5, -18),   # right cheek lower
    (0.42, 0.78, 0.12, 25,  12),   # chin left
    (0.62, 0.76, 0.09, 1,  -8),   # chin right
]

# ── Freckle config ────────────────────────────────────────────────────────────
# Fixed freckle positions as (fx, fy) fractions of face box
_FRECKLE_POSITIONS: List[Tuple[float, float]] = []

# Cache: original BGRA + one recoloured BGRA per colour
_star_base_cache:   Optional[np.ndarray]              = None
_star_colour_cache: Optional[List[np.ndarray]]        = None


# ─────────────────────────────────────────────────────────────────────────────
def _build_star_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Extract star shape as an alpha mask from the blue star PNG.
    Background is a light grey checkerboard (~175-195 BGR).
    Star is blue (HSV hue 85-130, sat > 60).
    Returns uint8 alpha channel same size as input.
    """
    hsv: np.ndarray      = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue: np.ndarray      = hsv[:, :, 0].astype(np.float32)
    sat: np.ndarray      = hsv[:, :, 1].astype(np.float32)

    hue_mask: np.ndarray = (
        np.clip((hue - 80.0) / 8.0, 0.0, 1.0) *
        np.clip((135.0 - hue) / 8.0, 0.0, 1.0)
    )
    sat_mask: np.ndarray = np.clip((sat - 50.0) / 40.0, 0.0, 1.0)

    alpha_f: np.ndarray  = hue_mask * sat_mask * 255.0
    alpha: np.ndarray    = np.clip(alpha_f, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), sigmaX=1.5)
    return alpha


def _make_coloured_star(bgr: np.ndarray, alpha: np.ndarray,
                        colour_bgr: Tuple[int, int, int]) -> np.ndarray:
    """
    Recolour the star to a solid colour while keeping its alpha shape.
    Returns a BGRA uint8 image.
    """
    h, w = bgr.shape[:2]
    coloured: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
    coloured[:] = colour_bgr

    # Blend original texture slightly into the flat colour for a natural look
    # 70% flat colour + 30% luminance texture from original
    gray: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lum: np.ndarray  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    col: np.ndarray  = coloured.astype(np.float32)
    mixed: np.ndarray = np.clip(col * 0.70 + lum * 0.30, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(mixed)
    return cv2.merge([b, g, r, alpha])


def _load_star_variants() -> List[np.ndarray]:
    """
    Load star.png once, extract alpha, generate one BGRA variant per colour.
    Cached after first call.
    """
    global _star_base_cache, _star_colour_cache
    if _star_colour_cache is not None:
        return _star_colour_cache

    bgr: np.ndarray = cv2.imread(_ASSET_PATH, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(
            "star.png not found at: {}\n"
            "Place the file at  assets/star.png  relative to your project root."
            .format(_ASSET_PATH)
        )

    alpha: np.ndarray = _build_star_mask(bgr)
    _star_colour_cache = [
        _make_coloured_star(bgr, alpha, c) for c in _STAR_COLOURS
    ]
    return _star_colour_cache


# ─────────────────────────────────────────────────────────────────────────────
def _overlay_bgra(
    base:    np.ndarray,
    overlay: np.ndarray,
    cx:      int,
    cy:      int,
    size:    int,
    angle:   float,
    opacity: float,
) -> np.ndarray:
    """
    Resize, rotate and alpha-composite a BGRA overlay centred at (cx, cy).
    """
    if size < 4:
        return base

    # Resize
    resized: np.ndarray = cv2.resize(
        overlay, (size, size), interpolation=cv2.INTER_AREA
    )

    # Rotate around centre
    if abs(angle) > 0.5:
        M: np.ndarray = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        resized = cv2.warpAffine(
            resized, M, (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    fh: int = base.shape[0]
    fw: int = base.shape[1]
    x: int  = cx - size // 2
    y: int  = cy - size // 2

    dst_x1: int = max(x, 0);       dst_y1: int = max(y, 0)
    dst_x2: int = min(x + size, fw); dst_y2: int = min(y + size, fh)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return base

    src_x1: int = dst_x1 - x;  src_y1: int = dst_y1 - y
    src_x2: int = src_x1 + (dst_x2 - dst_x1)
    src_y2: int = src_y1 + (dst_y2 - dst_y1)

    result: np.ndarray = base.copy()
    roi:    np.ndarray = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    patch:  np.ndarray = resized[src_y1:src_y2, src_x1:src_x2]

    patch_bgr: np.ndarray = patch[:, :, :3].astype(np.float32)
    patch_a:   np.ndarray = patch[:, :, 3:4].astype(np.float32) / 255.0 * float(opacity)

    blended: np.ndarray = roi * (1.0 - patch_a) + patch_bgr * patch_a
    result[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _apply_freckles(
    img: np.ndarray,
    fx: int, fy: int, fw: int, fh: int,
) -> np.ndarray:
    """
    Draw soft brown freckle dots scattered across the nose/cheek area.
    Fixed positions so they don't flicker between frames.
    """
    result: np.ndarray = img.copy()
    freckle_colour: Tuple[int, int, int] = (55, 80, 120)  # warm brown BGR
    freckle_r: int = max(2, int(fw * 0.012))              # radius scales with face

    overlay: np.ndarray = result.copy()
    for (pfx, pfy) in _FRECKLE_POSITIONS:
        cx: int = fx + int(fw * pfx)
        cy: int = fy + int(fh * pfy)
        cv2.circle(overlay, (cx, cy), freckle_r, freckle_colour, -1)
        # Slightly larger faint outer ring for softness
        cv2.circle(overlay, (cx, cy), freckle_r + 1, freckle_colour, 1)

    # Blend freckles at low opacity — subtle, natural
    result = cv2.addWeighted(result, 0.72, overlay, 0.28, 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
def apply_star_face(img: np.ndarray, star_opacity: float = 0.92) -> np.ndarray:
    """
    Star Face filter — scatters coloured star stickers across the face,
    adds a medium pink tint, and draws soft freckles on the nose/cheeks.

    Pipeline:
      1. Detect largest face with Haar cascade.
      2. Apply medium pink tint inside face box.
      3. Draw freckles on nose + cheek region.
      4. Load star.png → generate 7 recoloured BGRA variants (cached).
      5. Place 12 stars at fixed anatomical positions (no flicker).
         Each star has a unique: colour, size, rotation angle.

    Stars are placed at FIXED fractional positions within the face box
    so they sit still on the face — no time-based animation.

    Args:
        img          : BGR uint8 image (H x W x 3)
        star_opacity : star sticker opacity 0.0-1.0 (default 0.92 = crisp)

    Returns:
        BGR uint8 image with stars, pink tint, and freckles applied.
        Returns copy of original if no face detected.
    """
    face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(_FACE_XML)
    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    raw_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if not isinstance(raw_faces, np.ndarray) or len(raw_faces) == 0:
        return img.copy()

    face_box: list = max(raw_faces.tolist(), key=lambda f: f[2] * f[3])
    fx: int = int(face_box[0])
    fy: int = int(face_box[1])
    fw: int = int(face_box[2])
    fh: int = int(face_box[3])

    # ── 1. Freckles ───────────────────────────────────────────────────────────
    result: np.ndarray = _apply_freckles(img, fx, fy, fw, fh)

    # ── 3. Load star variants ─────────────────────────────────────────────────
    star_variants: List[np.ndarray] = _load_star_variants()

    # ── 4. Place stars at fixed positions ────────────────────────────────────
    for (pfx, pfy, size_frac, colour_idx, rotation) in _STAR_PLACEMENTS:
        cx: int   = fx + int(fw * pfx)
        cy: int   = fy + int(fh * pfy)
        size: int = max(8, int(fw * size_frac))
        variant   = star_variants[colour_idx % len(star_variants)]

        result = _overlay_bgra(
            result, variant, cx, cy, size, rotation, star_opacity
        )

    return result