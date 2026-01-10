"""
Quad-tree segmentation for adaptive block sizing.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .config import PredictionMethod


@dataclass
class Segment:
    """Represents a segment (block) in the image."""
    x: int
    y: int
    width: int
    height: int
    prediction_method: PredictionMethod = PredictionMethod.PAETH

    def __repr__(self) -> str:
        return f"Segment({self.x}, {self.y}, {self.width}x{self.height})"


def calculate_std(plane: np.ndarray, x: int, y: int, width: int, height: int) -> float:
    """Calculate standard deviation of a block."""
    if width <= 0 or height <= 0:
        return 0.0

    # Extract block with bounds checking
    x_end = min(x + width, plane.shape[1])
    y_end = min(y + height, plane.shape[0])

    if x >= plane.shape[1] or y >= plane.shape[0]:
        return 0.0

    block = plane[y:y_end, x:x_end]
    if block.size == 0:
        return 0.0

    return float(np.std(block))


def segment_recursive(plane: np.ndarray, x: int, y: int, width: int, height: int,
                      min_size: int, max_size: int, precision: float,
                      prediction_method: PredictionMethod) -> List[Segment]:
    """Recursively segment a block using quad-tree decomposition."""
    segments = []

    # Check if we should split
    should_split = False

    if width > min_size and height > min_size:
        std = calculate_std(plane, x, y, width, height)
        if std > precision and (width > min_size * 2 or height > min_size * 2):
            should_split = True

    # Don't split if we're at minimum size or block is too small
    if width <= min_size or height <= min_size:
        should_split = False

    # Don't split if we're at maximum size and block is homogeneous enough
    if width <= max_size and height <= max_size and not should_split:
        segments.append(Segment(x, y, width, height, prediction_method))
        return segments

    if should_split:
        half_w = width // 2
        half_h = height // 2

        # Make sure we don't create blocks smaller than min_size
        if half_w >= min_size and half_h >= min_size:
            # Top-left
            segments.extend(segment_recursive(
                plane, x, y, half_w, half_h,
                min_size, max_size, precision, prediction_method
            ))
            # Top-right
            segments.extend(segment_recursive(
                plane, x + half_w, y, width - half_w, half_h,
                min_size, max_size, precision, prediction_method
            ))
            # Bottom-left
            segments.extend(segment_recursive(
                plane, x, y + half_h, half_w, height - half_h,
                min_size, max_size, precision, prediction_method
            ))
            # Bottom-right
            segments.extend(segment_recursive(
                plane, x + half_w, y + half_h, width - half_w, height - half_h,
                min_size, max_size, precision, prediction_method
            ))
        else:
            segments.append(Segment(x, y, width, height, prediction_method))
    else:
        segments.append(Segment(x, y, width, height, prediction_method))

    return segments


def segment_plane(plane: np.ndarray, min_size: int = 2, max_size: int = 256,
                  precision: float = 15.0,
                  prediction_method: PredictionMethod = PredictionMethod.PAETH) -> List[Segment]:
    """
    Segment an image plane using quad-tree decomposition.

    Args:
        plane: 2D numpy array representing a single color channel
        min_size: Minimum block size (default 2)
        max_size: Maximum block size (default 256)
        precision: Segmentation threshold based on standard deviation (default 15.0)
        prediction_method: Default prediction method for segments

    Returns:
        List of Segment objects representing the quad-tree decomposition
    """
    height, width = plane.shape

    # Ensure dimensions are valid
    if width < min_size or height < min_size:
        return [Segment(0, 0, width, height, prediction_method)]

    # Start recursive segmentation
    segments = segment_recursive(
        plane, 0, 0, width, height,
        min_size, max_size, precision, prediction_method
    )

    return segments


def next_power_of_2(n: int) -> int:
    """Find the next power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def pad_to_power_of_2(plane: np.ndarray, ref_value: int = 128) -> Tuple[np.ndarray, int, int]:
    """
    Pad a plane to power-of-2 dimensions.

    Returns:
        Tuple of (padded_plane, original_width, original_height)
    """
    orig_height, orig_width = plane.shape
    new_width = next_power_of_2(orig_width)
    new_height = next_power_of_2(orig_height)

    if new_width == orig_width and new_height == orig_height:
        return plane, orig_width, orig_height

    padded = np.full((new_height, new_width), ref_value, dtype=plane.dtype)
    padded[:orig_height, :orig_width] = plane

    return padded, orig_width, orig_height


def unpad(plane: np.ndarray, orig_width: int, orig_height: int) -> np.ndarray:
    """Remove padding from a plane."""
    return plane[:orig_height, :orig_width]


def sort_segments_raster(segments: List[Segment]) -> List[Segment]:
    """Sort segments in raster scan order (top-to-bottom, left-to-right)."""
    return sorted(segments, key=lambda s: (s.y, s.x))


def get_segment_at(segments: List[Segment], x: int, y: int) -> Segment:
    """Find the segment containing the given pixel coordinates."""
    for seg in segments:
        if (seg.x <= x < seg.x + seg.width and
            seg.y <= y < seg.y + seg.height):
            return seg
    return None


def count_segments_by_size(segments: List[Segment]) -> dict:
    """Count segments grouped by their size."""
    counts = {}
    for seg in segments:
        size = (seg.width, seg.height)
        counts[size] = counts.get(size, 0) + 1
    return counts
