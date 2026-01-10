"""
Image plane management.
Handles color channel separation, padding, and conversion.
"""

import numpy as np
from typing import Tuple, Optional
from .config import ColorSpace
from .colorspaces import convert_to_colorspace, convert_from_colorspace
from .segment import next_power_of_2


class Planes:
    """
    Manages 3 color channel planes for image processing.
    """

    def __init__(self, width: int, height: int, ref_color: Tuple[int, int, int] = (128, 128, 128)):
        """
        Initialize planes with given dimensions.

        Args:
            width: Image width
            height: Image height
            ref_color: Reference color for padding (R, G, B)
        """
        self.original_width = width
        self.original_height = height
        self.ref_color = ref_color

        # Pad to power of 2
        self.width = next_power_of_2(width)
        self.height = next_power_of_2(height)

        # Initialize 3 planes
        self.planes = [
            np.full((self.height, self.width), ref_color[i], dtype=np.int32)
            for i in range(3)
        ]

    @classmethod
    def from_image(cls, image: np.ndarray, color_space: ColorSpace = ColorSpace.RGB,
                   ref_color: Tuple[int, int, int] = (128, 128, 128)) -> "Planes":
        """
        Create Planes from an image array.

        Args:
            image: RGB image array (H, W, 3) or (H, W, 4)
            color_space: Target color space for processing
            ref_color: Reference/border color
        """
        height, width = image.shape[:2]
        planes = cls(width, height, ref_color)

        # Extract RGB channels
        for y in range(height):
            for x in range(width):
                r, g, b = int(image[y, x, 0]), int(image[y, x, 1]), int(image[y, x, 2])

                # Convert to target color space
                c0, c1, c2 = convert_to_colorspace(r, g, b, color_space)

                planes.planes[0][y, x] = c0
                planes.planes[1][y, x] = c1
                planes.planes[2][y, x] = c2

        return planes

    def to_image(self, color_space: ColorSpace = ColorSpace.RGB) -> np.ndarray:
        """
        Convert planes back to RGB image.

        Args:
            color_space: Source color space of the planes

        Returns:
            RGB image array (H, W, 3)
        """
        image = np.zeros((self.original_height, self.original_width, 3), dtype=np.uint8)

        for y in range(self.original_height):
            for x in range(self.original_width):
                c0 = int(self.planes[0][y, x])
                c1 = int(self.planes[1][y, x])
                c2 = int(self.planes[2][y, x])

                # Convert back to RGB
                r, g, b = convert_from_colorspace(c0, c1, c2, color_space)

                image[y, x, 0] = max(0, min(255, r))
                image[y, x, 1] = max(0, min(255, g))
                image[y, x, 2] = max(0, min(255, b))

        return image

    def get_plane(self, channel: int) -> np.ndarray:
        """Get a single color plane."""
        return self.planes[channel]

    def set_plane(self, channel: int, data: np.ndarray) -> None:
        """Set a single color plane."""
        self.planes[channel] = data.astype(np.int32)

    def get_value(self, channel: int, x: int, y: int) -> int:
        """Get value at position, returning ref value if out of bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.planes[channel][y, x])
        return self.ref_color[channel]

    def set_value(self, channel: int, x: int, y: int, value: int) -> None:
        """Set value at position if within bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.planes[channel][y, x] = value

    def get_ref_value(self, channel: int) -> int:
        """Get reference value for a channel."""
        return self.ref_color[channel]

    def get_original_size(self) -> Tuple[int, int]:
        """Get original image dimensions."""
        return (self.original_width, self.original_height)

    def get_padded_size(self) -> Tuple[int, int]:
        """Get padded dimensions."""
        return (self.width, self.height)

    def copy(self) -> "Planes":
        """Create a deep copy."""
        new_planes = Planes(self.original_width, self.original_height, self.ref_color)
        for i in range(3):
            new_planes.planes[i] = self.planes[i].copy()
        return new_planes

    def crop_to_original(self) -> None:
        """Crop planes to original size (in-place)."""
        for i in range(3):
            self.planes[i] = self.planes[i][:self.original_height, :self.original_width]
        self.width = self.original_width
        self.height = self.original_height


def load_image(path: str) -> np.ndarray:
    """
    Load an image from file.

    Args:
        path: Path to image file (PNG, JPG, BMP)

    Returns:
        RGB image array (H, W, 3)
    """
    try:
        from PIL import Image
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except ImportError:
        # Fallback to imageio if PIL not available
        import imageio
        img = imageio.imread(path)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        return img


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save an image to file.

    Args:
        image: RGB image array (H, W, 3)
        path: Output path (PNG format recommended)
    """
    try:
        from PIL import Image
        img = Image.fromarray(image.astype(np.uint8))
        img.save(path)
    except ImportError:
        # Fallback to imageio if PIL not available
        import imageio
        imageio.imwrite(path, image.astype(np.uint8))


def image_from_planes(planes: Planes, color_space: ColorSpace) -> np.ndarray:
    """Convert Planes to RGB image array."""
    return planes.to_image(color_space)


def planes_from_image(image: np.ndarray, color_space: ColorSpace,
                      ref_color: Tuple[int, int, int] = (128, 128, 128)) -> Planes:
    """Create Planes from RGB image array."""
    return Planes.from_image(image, color_space, ref_color)
