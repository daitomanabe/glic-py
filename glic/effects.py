"""
Post-processing effects for decoded images.
Includes 6 artistic effect types.
"""

import random
import numpy as np
from typing import Callable
from .config import EffectType, EffectConfig


def clamp_uint8(value: float) -> int:
    """Clamp value to 0-255 range."""
    return int(max(0, min(255, value)))


# ============ Pixelate Effect ============

def effect_pixelate(image: np.ndarray, config: EffectConfig) -> np.ndarray:
    """
    Apply pixelation/mosaic effect.

    Args:
        image: RGB image (H, W, 3)
        config: Effect configuration with block_size and intensity
    """
    result = image.copy()
    height, width = image.shape[:2]

    block_size = max(1, config.block_size * config.intensity // 100)
    if block_size < 2:
        return result

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Calculate block bounds
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)

            # Get average color of block
            block = image[y:y_end, x:x_end]
            avg_color = np.mean(block, axis=(0, 1))

            # Fill block with average color
            result[y:y_end, x:x_end] = avg_color

    return result


# ============ Scanline Effect ============

def effect_scanline(image: np.ndarray, config: EffectConfig) -> np.ndarray:
    """
    Apply CRT monitor scanline effect.

    Args:
        image: RGB image (H, W, 3)
        config: Effect configuration with intensity and offset
    """
    result = image.copy().astype(np.float64)
    height, width = image.shape[:2]

    # Scanline intensity (0-1)
    intensity = config.intensity / 100.0
    line_spacing = max(1, config.block_size)
    offset = config.offset

    for y in range(height):
        if (y + offset) % line_spacing < line_spacing // 2:
            # Darken scanline rows
            result[y, :] *= (1.0 - intensity * 0.5)

    return np.clip(result, 0, 255).astype(np.uint8)


# ============ Chromatic Aberration Effect ============

def effect_chromatic(image: np.ndarray, config: EffectConfig) -> np.ndarray:
    """
    Apply chromatic aberration (RGB channel separation).

    Args:
        image: RGB image (H, W, 3)
        config: Effect configuration with intensity (determines shift amount)
    """
    result = np.zeros_like(image)
    height, width = image.shape[:2]

    # Shift amount based on intensity
    shift = max(1, config.intensity * config.block_size // 100)

    # Red channel shifted left
    if shift < width:
        result[:, :-shift, 0] = image[:, shift:, 0]
        result[:, -shift:, 0] = image[:, -1:, 0]

    # Green channel unchanged
    result[:, :, 1] = image[:, :, 1]

    # Blue channel shifted right
    if shift < width:
        result[:, shift:, 2] = image[:, :-shift, 2]
        result[:, :shift, 2] = image[:, :1, 2]

    return result


# ============ Dither Effect ============

# Bayer 4x4 dithering matrix
BAYER_4X4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
], dtype=np.float64) / 16.0


def effect_dither(image: np.ndarray, config: EffectConfig) -> np.ndarray:
    """
    Apply Bayer pattern dithering effect.

    Args:
        image: RGB image (H, W, 3)
        config: Effect configuration with intensity and levels
    """
    result = image.copy().astype(np.float64)
    height, width = image.shape[:2]

    # Number of color levels
    levels = max(2, config.levels)
    intensity = config.intensity / 100.0

    # Apply dithering
    for y in range(height):
        for x in range(width):
            threshold = BAYER_4X4[y % 4, x % 4]

            for c in range(3):
                val = result[y, x, c] / 255.0

                # Quantize with dithering
                quantized = np.floor(val * (levels - 1) + threshold * intensity) / (levels - 1)
                result[y, x, c] = quantized * 255

    return np.clip(result, 0, 255).astype(np.uint8)


# ============ Posterize Effect ============

def effect_posterize(image: np.ndarray, config: EffectConfig) -> np.ndarray:
    """
    Apply posterization (color level reduction).

    Args:
        image: RGB image (H, W, 3)
        config: Effect configuration with levels
    """
    result = image.copy().astype(np.float64)

    # Number of levels per channel
    levels = max(2, config.levels)

    # Quantize to levels
    scale = 255.0 / (levels - 1)
    result = np.floor(result / scale + 0.5) * scale

    return np.clip(result, 0, 255).astype(np.uint8)


# ============ Glitch Shift Effect ============

def effect_glitch_shift(image: np.ndarray, config: EffectConfig) -> np.ndarray:
    """
    Apply random row displacement (glitch) effect.

    Args:
        image: RGB image (H, W, 3)
        config: Effect configuration with intensity and block_size
    """
    result = image.copy()
    height, width = image.shape[:2]

    # Random seed for reproducibility based on config
    random.seed(config.offset)

    # Number of glitch blocks
    num_glitches = max(1, config.intensity * height // 1000)

    for _ in range(num_glitches):
        # Random row range
        start_y = random.randint(0, height - 1)
        block_h = random.randint(1, min(config.block_size, height - start_y))

        # Random horizontal shift
        max_shift = width * config.intensity // 100
        shift = random.randint(-max_shift, max_shift)

        # Apply shift to rows
        for y in range(start_y, min(start_y + block_h, height)):
            if shift > 0:
                result[y, shift:] = image[y, :-shift]
                result[y, :shift] = image[y, -shift:]
            elif shift < 0:
                result[y, :shift] = image[y, -shift:]
                result[y, shift:] = image[y, :-shift]

    return result


# ============ Effect Dispatcher ============

_EFFECTS = {
    EffectType.PIXELATE: effect_pixelate,
    EffectType.SCANLINE: effect_scanline,
    EffectType.CHROMATIC: effect_chromatic,
    EffectType.DITHER: effect_dither,
    EffectType.POSTERIZE: effect_posterize,
    EffectType.GLITCH_SHIFT: effect_glitch_shift,
}


def apply_effect(image: np.ndarray, config: EffectConfig) -> np.ndarray:
    """Apply a single effect to an image."""
    effect_func = _EFFECTS.get(config.effect_type)
    if effect_func is None:
        return image
    return effect_func(image, config)


def apply_effects(image: np.ndarray, configs: list) -> np.ndarray:
    """Apply multiple effects in sequence."""
    result = image.copy()
    for config in configs:
        result = apply_effect(result, config)
    return result


def get_available_effects() -> list:
    """Get list of available effect types."""
    return list(EffectType)


def create_effect_config(effect_type: EffectType, intensity: int = 50,
                         block_size: int = 8, offset: int = 0,
                         levels: int = 4) -> EffectConfig:
    """Create an effect configuration."""
    return EffectConfig(
        effect_type=effect_type,
        intensity=intensity,
        block_size=block_size,
        offset=offset,
        levels=levels
    )
