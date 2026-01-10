"""
Configuration enums and data classes for GLIC codec.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple


class ColorSpace(Enum):
    """Supported color spaces for image processing."""
    RGB = auto()
    HSB = auto()
    HWB = auto()
    OHTA = auto()
    CMY = auto()
    XYZ = auto()
    YXY = auto()
    LAB = auto()
    LUV = auto()
    HCL = auto()
    YUV = auto()
    YPbPr = auto()
    YCbCr = auto()
    YDbDr = auto()
    GS = auto()  # Grayscale
    RGGBG = auto()  # R-G, G, B-G


class PredictionMethod(Enum):
    """Prediction algorithms for residual calculation."""
    # Original GLIC methods
    NONE = auto()
    CORNER = auto()
    H = auto()  # Horizontal
    V = auto()  # Vertical
    DC = auto()  # DC average
    DCMEDIAN = auto()
    MEDIAN = auto()
    AVG = auto()
    TRUEMOTION = auto()
    PAETH = auto()
    LDIAG = auto()  # Left diagonal
    HV = auto()  # Horizontal-Vertical
    JPEGLS = auto()
    DIFF = auto()
    REF = auto()
    ANGLE = auto()
    # New C++ methods
    SPIRAL = auto()
    NOISE = auto()
    GRADIENT = auto()
    MIRROR = auto()
    WAVE = auto()
    CHECKERBOARD = auto()
    RADIAL = auto()
    EDGE = auto()
    # Meta predictors
    SAD = auto()  # Sum of Absolute Differences
    BSAD = auto()  # Block SAD
    RANDOM = auto()


class EncodingMethod(Enum):
    """Encoding methods for compressed data."""
    RAW = auto()
    PACKED = auto()
    RLE = auto()  # Run-Length Encoding
    DELTA = auto()
    XOR = auto()
    ZIGZAG = auto()


class TransformType(Enum):
    """Wavelet transform types."""
    NONE = auto()
    FWT = auto()  # Fast Wavelet Transform
    WPT = auto()  # Wavelet Packet Transform


class WaveletType(Enum):
    """Supported wavelet families."""
    HAAR = auto()
    # Daubechies
    DB2 = auto()
    DB3 = auto()
    DB4 = auto()
    DB5 = auto()
    DB6 = auto()
    DB7 = auto()
    DB8 = auto()
    DB9 = auto()
    DB10 = auto()
    # Symlet
    SYM2 = auto()
    SYM3 = auto()
    SYM4 = auto()
    SYM5 = auto()
    SYM6 = auto()
    SYM7 = auto()
    SYM8 = auto()
    SYM9 = auto()
    SYM10 = auto()
    # Coiflet
    COIF1 = auto()
    COIF2 = auto()
    COIF3 = auto()
    COIF4 = auto()
    COIF5 = auto()
    # Biorthogonal
    BIOR11 = auto()
    BIOR13 = auto()
    BIOR15 = auto()
    BIOR22 = auto()
    BIOR24 = auto()
    BIOR26 = auto()
    BIOR28 = auto()
    BIOR31 = auto()
    BIOR33 = auto()
    BIOR35 = auto()
    BIOR37 = auto()
    BIOR39 = auto()
    BIOR44 = auto()


class ClampMethod(Enum):
    """Clamping methods for values."""
    NONE = auto()
    CLAMP = auto()
    WRAP = auto()


class EffectType(Enum):
    """Post-processing effect types."""
    PIXELATE = auto()
    SCANLINE = auto()
    CHROMATIC = auto()
    DITHER = auto()
    POSTERIZE = auto()
    GLITCH_SHIFT = auto()


@dataclass
class ChannelConfig:
    """Configuration for a single color channel."""
    min_block_size: int = 2
    max_block_size: int = 256
    segmentation_precision: float = 15.0
    prediction_method: PredictionMethod = PredictionMethod.PAETH
    quantization_value: int = 110
    clamp_method: ClampMethod = ClampMethod.NONE
    transform_type: TransformType = TransformType.FWT
    wavelet_type: WaveletType = WaveletType.SYM8
    transform_scale: int = 20
    encoding_method: EncodingMethod = EncodingMethod.PACKED


@dataclass
class EffectConfig:
    """Configuration for a post-processing effect."""
    effect_type: EffectType = EffectType.PIXELATE
    intensity: int = 50
    block_size: int = 8
    offset: int = 0
    levels: int = 4


@dataclass
class CodecConfig:
    """Main codec configuration."""
    color_space: ColorSpace = ColorSpace.HWB
    border_color: Tuple[int, int, int] = (128, 128, 128)
    channels: List[ChannelConfig] = field(default_factory=lambda: [
        ChannelConfig(),
        ChannelConfig(),
        ChannelConfig(),
    ])
    effects: List[EffectConfig] = field(default_factory=list)

    def copy(self) -> "CodecConfig":
        """Create a deep copy of the configuration."""
        import copy
        return copy.deepcopy(self)


# Magic number for GLIC file format
GLIC_MAGIC = 0x474C4332  # "GLC2"
GLIC_VERSION = 1
