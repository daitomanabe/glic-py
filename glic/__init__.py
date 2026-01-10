"""
GLIC - GLitch Image Codec
A Python implementation of the artistic glitch image codec.

Quick Start:
    >>> import glic
    >>> glic.glitch("input.png", "output.png")
    >>> glic.glitch("input.png", intensity=0.8, preset="heavy_glitch")
    >>> glic.batch_glitch("input_folder/", "output_folder/")

For more control:
    >>> from glic import GlicCodec, CodecConfig
    >>> config = CodecConfig()
    >>> codec = GlicCodec(config)
    >>> codec.encode_decode("input.png", "output.png")
"""

from .codec import GlicCodec
from .config import (
    ColorSpace,
    PredictionMethod,
    EncodingMethod,
    TransformType,
    WaveletType,
    ClampMethod,
    EffectType,
    ChannelConfig,
    CodecConfig,
    EffectConfig,
)

# Easy-to-use API
from .easy import (
    glitch,
    glitch_image,
    batch_glitch,
    random_glitch,
    create_variations,
    list_presets,
    list_colorspaces,
    list_predictions,
    list_effects,
    demo,
)

__version__ = "1.0.0"
__all__ = [
    # Easy API (most common use)
    "glitch",
    "glitch_image",
    "batch_glitch",
    "random_glitch",
    "create_variations",
    "list_presets",
    "list_colorspaces",
    "list_predictions",
    "list_effects",
    "demo",
    # Advanced API
    "GlicCodec",
    "ColorSpace",
    "PredictionMethod",
    "EncodingMethod",
    "TransformType",
    "WaveletType",
    "ClampMethod",
    "EffectType",
    "ChannelConfig",
    "CodecConfig",
    "EffectConfig",
]
