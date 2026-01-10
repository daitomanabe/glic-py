"""
Prediction algorithms for residual calculation.
Includes 24 different prediction methods for creating compression artifacts.
"""

import math
import random
from typing import Callable, List, Tuple, Optional
import numpy as np
from .config import PredictionMethod


def get_ref_value(plane: np.ndarray, x: int, y: int, ref_value: int) -> int:
    """Get pixel value, returning reference value if out of bounds."""
    if 0 <= x < plane.shape[1] and 0 <= y < plane.shape[0]:
        return int(plane[y, x])
    return ref_value


def hash_position(x: int, y: int) -> int:
    """Simple hash function for position-based noise."""
    h = x * 374761393 + y * 668265263
    h = (h ^ (h >> 13)) * 1274126177
    return h & 0xFFFFFFFF


# ============ Prediction Functions ============

def predict_none(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                 block_w: int, block_h: int, ref: int) -> int:
    """No prediction - returns reference value."""
    return ref


def predict_corner(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                   block_w: int, block_h: int, ref: int) -> int:
    """Use top-left corner of block."""
    return get_ref_value(plane, block_x - 1, block_y - 1, ref)


def predict_h(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
              block_w: int, block_h: int, ref: int) -> int:
    """Horizontal prediction - use left neighbor."""
    return get_ref_value(plane, x - 1, y, ref)


def predict_v(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
              block_w: int, block_h: int, ref: int) -> int:
    """Vertical prediction - use top neighbor."""
    return get_ref_value(plane, x, y - 1, ref)


def predict_dc(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
               block_w: int, block_h: int, ref: int) -> int:
    """DC prediction - average of top row and left column."""
    total = 0
    count = 0

    # Top row
    for i in range(block_w):
        total += get_ref_value(plane, block_x + i, block_y - 1, ref)
        count += 1

    # Left column
    for i in range(block_h):
        total += get_ref_value(plane, block_x - 1, block_y + i, ref)
        count += 1

    return total // count if count > 0 else ref


def predict_dcmedian(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                     block_w: int, block_h: int, ref: int) -> int:
    """DC prediction using median instead of mean."""
    values = []

    # Top row
    for i in range(block_w):
        values.append(get_ref_value(plane, block_x + i, block_y - 1, ref))

    # Left column
    for i in range(block_h):
        values.append(get_ref_value(plane, block_x - 1, block_y + i, ref))

    if not values:
        return ref

    values.sort()
    mid = len(values) // 2
    if len(values) % 2 == 0:
        return (values[mid - 1] + values[mid]) // 2
    return values[mid]


def predict_median(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                   block_w: int, block_h: int, ref: int) -> int:
    """Median of left, top, and top-left."""
    left = get_ref_value(plane, x - 1, y, ref)
    top = get_ref_value(plane, x, y - 1, ref)
    top_left = get_ref_value(plane, x - 1, y - 1, ref)

    values = [left, top, top_left]
    values.sort()
    return values[1]


def predict_avg(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                block_w: int, block_h: int, ref: int) -> int:
    """Average of left and top."""
    left = get_ref_value(plane, x - 1, y, ref)
    top = get_ref_value(plane, x, y - 1, ref)
    return (left + top) // 2


def predict_truemotion(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                       block_w: int, block_h: int, ref: int) -> int:
    """TrueMotion prediction: left + top - top_left."""
    left = get_ref_value(plane, x - 1, y, ref)
    top = get_ref_value(plane, x, y - 1, ref)
    top_left = get_ref_value(plane, x - 1, y - 1, ref)
    return max(0, min(255, left + top - top_left))


def predict_paeth(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                  block_w: int, block_h: int, ref: int) -> int:
    """Paeth prediction (PNG predictor)."""
    left = get_ref_value(plane, x - 1, y, ref)
    top = get_ref_value(plane, x, y - 1, ref)
    top_left = get_ref_value(plane, x - 1, y - 1, ref)

    p = left + top - top_left
    pa = abs(p - left)
    pb = abs(p - top)
    pc = abs(p - top_left)

    if pa <= pb and pa <= pc:
        return left
    elif pb <= pc:
        return top
    return top_left


def predict_ldiag(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                  block_w: int, block_h: int, ref: int) -> int:
    """Left diagonal prediction."""
    return get_ref_value(plane, x - 1, y - 1, ref)


def predict_hv(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
               block_w: int, block_h: int, ref: int) -> int:
    """HV prediction - choose between H and V based on gradients."""
    left = get_ref_value(plane, x - 1, y, ref)
    top = get_ref_value(plane, x, y - 1, ref)
    top_left = get_ref_value(plane, x - 1, y - 1, ref)

    h_grad = abs(left - top_left)
    v_grad = abs(top - top_left)

    if h_grad < v_grad:
        return left
    return top


def predict_jpegls(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                   block_w: int, block_h: int, ref: int) -> int:
    """JPEG-LS prediction."""
    left = get_ref_value(plane, x - 1, y, ref)
    top = get_ref_value(plane, x, y - 1, ref)
    top_left = get_ref_value(plane, x - 1, y - 1, ref)

    min_val = min(left, top)
    max_val = max(left, top)

    if top_left >= max_val:
        return min_val
    elif top_left <= min_val:
        return max_val
    return left + top - top_left


def predict_diff(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                 block_w: int, block_h: int, ref: int) -> int:
    """Difference prediction."""
    left = get_ref_value(plane, x - 1, y, ref)
    top = get_ref_value(plane, x, y - 1, ref)
    return max(0, min(255, (left + top) // 2))


def predict_ref(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                block_w: int, block_h: int, ref: int) -> int:
    """Reference value prediction."""
    return ref


def predict_angle(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                  block_w: int, block_h: int, ref: int) -> int:
    """Angle-based prediction."""
    dx = x - block_x
    dy = y - block_y

    if dx == 0 and dy == 0:
        return get_ref_value(plane, block_x - 1, block_y - 1, ref)

    angle = math.atan2(dy, dx)
    norm_angle = (angle + math.pi) / (2 * math.pi)

    left = get_ref_value(plane, x - 1, y, ref)
    top = get_ref_value(plane, x, y - 1, ref)
    top_left = get_ref_value(plane, x - 1, y - 1, ref)

    return int(left * (1 - norm_angle) + top * norm_angle)


# ============ New C++ Prediction Methods ============

def predict_spiral(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                   block_w: int, block_h: int, ref: int) -> int:
    """Spiral pattern prediction from center."""
    cx = block_x + block_w // 2
    cy = block_y + block_h // 2

    dx = x - cx
    dy = y - cy
    dist = math.sqrt(dx * dx + dy * dy)

    if dist == 0:
        return ref

    angle = math.atan2(dy, dx)
    spiral_offset = (dist + angle * 2) % block_w

    ref_x = block_x + int(spiral_offset) % block_w
    ref_y = block_y - 1

    return get_ref_value(plane, ref_x, ref_y, ref)


def predict_noise(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                  block_w: int, block_h: int, ref: int) -> int:
    """Hash-based pseudo-random prediction."""
    h = hash_position(x, y)
    return h % 256


def predict_gradient(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                     block_w: int, block_h: int, ref: int) -> int:
    """Bilinear interpolation from 4 corners."""
    # Get corner values
    tl = get_ref_value(plane, block_x - 1, block_y - 1, ref)
    tr = get_ref_value(plane, block_x + block_w, block_y - 1, ref)
    bl = get_ref_value(plane, block_x - 1, block_y + block_h, ref)
    br = get_ref_value(plane, block_x + block_w, block_y + block_h, ref)

    # Calculate interpolation factors
    fx = (x - block_x) / max(1, block_w - 1)
    fy = (y - block_y) / max(1, block_h - 1)

    # Bilinear interpolation
    top = tl * (1 - fx) + tr * fx
    bottom = bl * (1 - fx) + br * fx
    result = top * (1 - fy) + bottom * fy

    return int(max(0, min(255, result)))


def predict_mirror(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                   block_w: int, block_h: int, ref: int) -> int:
    """Mirror/reflection prediction."""
    dx = x - block_x
    dy = y - block_y

    # Mirror across top-left corner
    mirror_x = block_x - 1 - dx
    mirror_y = block_y - 1 - dy

    return get_ref_value(plane, mirror_x, mirror_y, ref)


def predict_wave(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                 block_w: int, block_h: int, ref: int) -> int:
    """Sine wave-based displacement prediction."""
    dx = x - block_x
    dy = y - block_y

    wave_x = math.sin(dx * 0.5) * 2
    wave_y = math.sin(dy * 0.5) * 2

    ref_x = int(x + wave_x) if x + wave_x >= 0 else x - 1
    ref_y = int(y + wave_y) if y + wave_y >= 0 else y - 1

    return get_ref_value(plane, ref_x, ref_y, ref)


def predict_checkerboard(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                         block_w: int, block_h: int, ref: int) -> int:
    """Alternating checkerboard prediction."""
    is_even = ((x - block_x) + (y - block_y)) % 2 == 0

    if is_even:
        return get_ref_value(plane, x - 1, y, ref)
    return get_ref_value(plane, x, y - 1, ref)


def predict_radial(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                   block_w: int, block_h: int, ref: int) -> int:
    """Radiating gradient from center."""
    cx = block_x + block_w // 2
    cy = block_y + block_h // 2

    dx = x - cx
    dy = y - cy
    max_dist = math.sqrt((block_w // 2) ** 2 + (block_h // 2) ** 2)
    dist = math.sqrt(dx * dx + dy * dy)

    if max_dist == 0:
        return ref

    factor = dist / max_dist

    # Get center reference
    center_ref = get_ref_value(plane, block_x + block_w // 2, block_y - 1, ref)
    edge_ref = get_ref_value(plane, block_x - 1, block_y - 1, ref)

    return int(center_ref * (1 - factor) + edge_ref * factor)


def predict_edge(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                 block_w: int, block_h: int, ref: int) -> int:
    """Edge detection-based prediction."""
    left = get_ref_value(plane, x - 1, y, ref)
    top = get_ref_value(plane, x, y - 1, ref)
    top_left = get_ref_value(plane, x - 1, y - 1, ref)
    top_right = get_ref_value(plane, x + 1, y - 1, ref)

    # Sobel-like edge detection
    gx = (top_right - top_left) // 2
    gy = (left - top) // 2

    edge_strength = abs(gx) + abs(gy)

    if edge_strength > 32:
        # Strong edge - use directional prediction
        if abs(gx) > abs(gy):
            return top
        return left

    return (left + top) // 2


# ============ Meta Predictors ============

_BASIC_METHODS = [
    PredictionMethod.NONE, PredictionMethod.CORNER, PredictionMethod.H,
    PredictionMethod.V, PredictionMethod.DC, PredictionMethod.DCMEDIAN,
    PredictionMethod.MEDIAN, PredictionMethod.AVG, PredictionMethod.TRUEMOTION,
    PredictionMethod.PAETH, PredictionMethod.LDIAG, PredictionMethod.HV,
    PredictionMethod.JPEGLS, PredictionMethod.DIFF, PredictionMethod.REF,
    PredictionMethod.ANGLE
]


def predict_sad(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                block_w: int, block_h: int, ref: int) -> int:
    """SAD (Sum of Absolute Differences) - try all and pick best."""
    actual = int(plane[y, x]) if 0 <= y < plane.shape[0] and 0 <= x < plane.shape[1] else ref

    best_pred = ref
    best_diff = float('inf')

    for method in _BASIC_METHODS[:8]:  # Try first 8 methods
        pred = get_predictor(method)(plane, x, y, block_x, block_y, block_w, block_h, ref)
        diff = abs(actual - pred)
        if diff < best_diff:
            best_diff = diff
            best_pred = pred

    return best_pred


def predict_bsad(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                 block_w: int, block_h: int, ref: int) -> int:
    """Block SAD - evaluate over entire block."""
    # For simplicity, just use PAETH as default
    return predict_paeth(plane, x, y, block_x, block_y, block_w, block_h, ref)


def predict_random(plane: np.ndarray, x: int, y: int, block_x: int, block_y: int,
                   block_w: int, block_h: int, ref: int) -> int:
    """Random prediction method selection."""
    method = random.choice(_BASIC_METHODS)
    return get_predictor(method)(plane, x, y, block_x, block_y, block_w, block_h, ref)


# ============ Predictor Registry ============

_PREDICTORS = {
    PredictionMethod.NONE: predict_none,
    PredictionMethod.CORNER: predict_corner,
    PredictionMethod.H: predict_h,
    PredictionMethod.V: predict_v,
    PredictionMethod.DC: predict_dc,
    PredictionMethod.DCMEDIAN: predict_dcmedian,
    PredictionMethod.MEDIAN: predict_median,
    PredictionMethod.AVG: predict_avg,
    PredictionMethod.TRUEMOTION: predict_truemotion,
    PredictionMethod.PAETH: predict_paeth,
    PredictionMethod.LDIAG: predict_ldiag,
    PredictionMethod.HV: predict_hv,
    PredictionMethod.JPEGLS: predict_jpegls,
    PredictionMethod.DIFF: predict_diff,
    PredictionMethod.REF: predict_ref,
    PredictionMethod.ANGLE: predict_angle,
    PredictionMethod.SPIRAL: predict_spiral,
    PredictionMethod.NOISE: predict_noise,
    PredictionMethod.GRADIENT: predict_gradient,
    PredictionMethod.MIRROR: predict_mirror,
    PredictionMethod.WAVE: predict_wave,
    PredictionMethod.CHECKERBOARD: predict_checkerboard,
    PredictionMethod.RADIAL: predict_radial,
    PredictionMethod.EDGE: predict_edge,
    PredictionMethod.SAD: predict_sad,
    PredictionMethod.BSAD: predict_bsad,
    PredictionMethod.RANDOM: predict_random,
}


def get_predictor(method: PredictionMethod) -> Callable:
    """Get the prediction function for the given method."""
    return _PREDICTORS.get(method, predict_paeth)


def apply_prediction(plane: np.ndarray, block_x: int, block_y: int,
                     block_w: int, block_h: int, ref: int,
                     method: PredictionMethod) -> np.ndarray:
    """Apply prediction to a block and return residuals."""
    predictor = get_predictor(method)
    residuals = np.zeros((block_h, block_w), dtype=np.int32)

    for dy in range(block_h):
        for dx in range(block_w):
            x = block_x + dx
            y = block_y + dy

            if 0 <= y < plane.shape[0] and 0 <= x < plane.shape[1]:
                predicted = predictor(plane, x, y, block_x, block_y, block_w, block_h, ref)
                actual = int(plane[y, x])
                residuals[dy, dx] = actual - predicted

    return residuals


def apply_inverse_prediction(residuals: np.ndarray, plane: np.ndarray,
                             block_x: int, block_y: int, ref: int,
                             method: PredictionMethod) -> None:
    """Apply inverse prediction to reconstruct pixel values in-place."""
    predictor = get_predictor(method)
    block_h, block_w = residuals.shape

    for dy in range(block_h):
        for dx in range(block_w):
            x = block_x + dx
            y = block_y + dy

            if 0 <= y < plane.shape[0] and 0 <= x < plane.shape[1]:
                predicted = predictor(plane, x, y, block_x, block_y, block_w, block_h, ref)
                value = predicted + residuals[dy, dx]
                plane[y, x] = max(0, min(255, value))


def calculate_sad(plane: np.ndarray, block_x: int, block_y: int,
                  block_w: int, block_h: int, ref: int,
                  method: PredictionMethod) -> int:
    """Calculate Sum of Absolute Differences for a prediction method."""
    residuals = apply_prediction(plane, block_x, block_y, block_w, block_h, ref, method)
    return int(np.sum(np.abs(residuals)))
