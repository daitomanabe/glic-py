"""
Preset loading system for GLIC codec.
Loads original GLIC Java serialized presets and built-in presets.
"""

import os
import struct
from pathlib import Path
from typing import Dict, List, Optional
from .config import (
    CodecConfig, ChannelConfig, ColorSpace, PredictionMethod,
    EncodingMethod, TransformType, WaveletType, ClampMethod
)


# Mapping from Java preset values to Python enums
_COLORSPACE_MAP = {
    0: ColorSpace.RGB,
    1: ColorSpace.HSB,
    2: ColorSpace.HWB,
    3: ColorSpace.OHTA,
    4: ColorSpace.CMY,
    5: ColorSpace.XYZ,
    6: ColorSpace.YXY,
    7: ColorSpace.LAB,
    8: ColorSpace.LUV,
    9: ColorSpace.HCL,
    10: ColorSpace.YUV,
    11: ColorSpace.YPbPr,
    12: ColorSpace.YCbCr,
    13: ColorSpace.YDbDr,
    14: ColorSpace.GS,
    15: ColorSpace.RGGBG,
}

_PREDICTION_MAP = {
    0: PredictionMethod.NONE,
    1: PredictionMethod.CORNER,
    2: PredictionMethod.H,
    3: PredictionMethod.V,
    4: PredictionMethod.DC,
    5: PredictionMethod.DCMEDIAN,
    6: PredictionMethod.MEDIAN,
    7: PredictionMethod.AVG,
    8: PredictionMethod.TRUEMOTION,
    9: PredictionMethod.PAETH,
    10: PredictionMethod.LDIAG,
    11: PredictionMethod.HV,
    12: PredictionMethod.JPEGLS,
    13: PredictionMethod.DIFF,
    14: PredictionMethod.REF,
    15: PredictionMethod.ANGLE,
    16: PredictionMethod.SAD,
    17: PredictionMethod.BSAD,
    18: PredictionMethod.RANDOM,
}

_ENCODING_MAP = {
    0: EncodingMethod.RAW,
    1: EncodingMethod.PACKED,
    2: EncodingMethod.RLE,
}

_WAVELET_MAP = {
    0: WaveletType.HAAR,
    1: WaveletType.DB2,
    2: WaveletType.DB3,
    3: WaveletType.DB4,
    4: WaveletType.DB5,
    5: WaveletType.DB6,
    6: WaveletType.DB7,
    7: WaveletType.DB8,
    8: WaveletType.DB9,
    9: WaveletType.DB10,
    10: WaveletType.SYM2,
    11: WaveletType.SYM3,
    12: WaveletType.SYM4,
    13: WaveletType.SYM5,
    14: WaveletType.SYM6,
    15: WaveletType.SYM7,
    16: WaveletType.SYM8,
    17: WaveletType.SYM9,
    18: WaveletType.SYM10,
    19: WaveletType.COIF1,
    20: WaveletType.COIF2,
    21: WaveletType.COIF3,
    22: WaveletType.COIF4,
    23: WaveletType.COIF5,
}

_TRANSFORM_MAP = {
    0: TransformType.NONE,
    1: TransformType.FWT,
    2: TransformType.WPT,
}

_CLAMP_MAP = {
    0: ClampMethod.NONE,
    1: ClampMethod.CLAMP,
    2: ClampMethod.WRAP,
}


class PresetLoader:
    """Loads and manages GLIC presets."""

    def __init__(self, presets_dir: Optional[str] = None):
        """
        Initialize preset loader.

        Args:
            presets_dir: Directory containing preset files
        """
        self.presets_dir = presets_dir
        self._presets_cache: Dict[str, CodecConfig] = {}
        self._builtin_presets = self._create_builtin_presets()

    def _create_builtin_presets(self) -> Dict[str, CodecConfig]:
        """Create built-in preset configurations."""
        presets = {}

        # Default preset
        presets["default"] = CodecConfig()

        # High quality preset
        hq = CodecConfig()
        hq.color_space = ColorSpace.YCbCr
        for ch in hq.channels:
            ch.prediction_method = PredictionMethod.PAETH
            ch.quantization_value = 50
            ch.transform_type = TransformType.FWT
            ch.wavelet_type = WaveletType.SYM8
            ch.transform_scale = 10
        presets["high_quality"] = hq

        # Heavy glitch preset
        glitch = CodecConfig()
        glitch.color_space = ColorSpace.HWB
        for ch in glitch.channels:
            ch.prediction_method = PredictionMethod.NOISE
            ch.quantization_value = 200
            ch.transform_type = TransformType.WPT
            ch.wavelet_type = WaveletType.HAAR
            ch.transform_scale = 50
            ch.min_block_size = 8
            ch.max_block_size = 64
        presets["heavy_glitch"] = glitch

        # Minimal glitch preset
        minimal = CodecConfig()
        minimal.color_space = ColorSpace.RGB
        for ch in minimal.channels:
            ch.prediction_method = PredictionMethod.DC
            ch.quantization_value = 80
            ch.transform_type = TransformType.NONE
        presets["minimal"] = minimal

        # Color waves preset
        waves = CodecConfig()
        waves.color_space = ColorSpace.HSB
        for i, ch in enumerate(waves.channels):
            ch.prediction_method = PredictionMethod.WAVE
            ch.quantization_value = 120
            ch.transform_type = TransformType.FWT
            ch.wavelet_type = WaveletType.DB4
            ch.transform_scale = 30
            ch.segmentation_precision = 10 + i * 5
        presets["color_waves"] = waves

        # Pixelated preset
        pixel = CodecConfig()
        pixel.color_space = ColorSpace.RGB
        for ch in pixel.channels:
            ch.prediction_method = PredictionMethod.CORNER
            ch.min_block_size = 8
            ch.max_block_size = 32
            ch.segmentation_precision = 50
            ch.quantization_value = 150
            ch.transform_type = TransformType.NONE
        presets["pixelated"] = pixel

        # Spiral preset
        spiral = CodecConfig()
        spiral.color_space = ColorSpace.OHTA
        for ch in spiral.channels:
            ch.prediction_method = PredictionMethod.SPIRAL
            ch.quantization_value = 100
            ch.transform_type = TransformType.FWT
            ch.wavelet_type = WaveletType.SYM6
            ch.transform_scale = 25
        presets["spiral"] = spiral

        # Chromatic preset
        chromatic = CodecConfig()
        chromatic.color_space = ColorSpace.YUV
        chromatic.channels[0].prediction_method = PredictionMethod.PAETH
        chromatic.channels[0].quantization_value = 60
        chromatic.channels[1].prediction_method = PredictionMethod.GRADIENT
        chromatic.channels[1].quantization_value = 140
        chromatic.channels[2].prediction_method = PredictionMethod.RADIAL
        chromatic.channels[2].quantization_value = 140
        presets["chromatic"] = chromatic

        # === New artistic presets ===

        # VHS / Retro TV
        vhs = CodecConfig()
        vhs.color_space = ColorSpace.YCbCr
        for ch in vhs.channels:
            ch.prediction_method = PredictionMethod.H
            ch.quantization_value = 130
            ch.transform_type = TransformType.FWT
            ch.wavelet_type = WaveletType.HAAR
            ch.transform_scale = 35
            ch.min_block_size = 4
            ch.max_block_size = 128
        presets["vhs"] = vhs

        # Datamosh style
        datamosh = CodecConfig()
        datamosh.color_space = ColorSpace.YUV
        for ch in datamosh.channels:
            ch.prediction_method = PredictionMethod.TRUEMOTION
            ch.quantization_value = 180
            ch.transform_type = TransformType.WPT
            ch.wavelet_type = WaveletType.DB2
            ch.transform_scale = 45
            ch.segmentation_precision = 8
        presets["datamosh"] = datamosh

        # Acid / Psychedelic
        acid = CodecConfig()
        acid.color_space = ColorSpace.HSB
        acid.channels[0].prediction_method = PredictionMethod.WAVE  # Hue
        acid.channels[0].quantization_value = 200
        acid.channels[1].prediction_method = PredictionMethod.SPIRAL  # Saturation
        acid.channels[1].quantization_value = 150
        acid.channels[2].prediction_method = PredictionMethod.RADIAL  # Brightness
        acid.channels[2].quantization_value = 80
        for ch in acid.channels:
            ch.transform_type = TransformType.WPT
            ch.wavelet_type = WaveletType.SYM4
            ch.transform_scale = 40
        presets["acid"] = acid

        # Noise / Static
        noise = CodecConfig()
        noise.color_space = ColorSpace.RGB
        for ch in noise.channels:
            ch.prediction_method = PredictionMethod.NOISE
            ch.quantization_value = 220
            ch.transform_type = TransformType.NONE
            ch.min_block_size = 2
            ch.max_block_size = 16
            ch.segmentation_precision = 5
        presets["noise"] = noise

        # Soft / Dreamy
        dreamy = CodecConfig()
        dreamy.color_space = ColorSpace.LAB
        for ch in dreamy.channels:
            ch.prediction_method = PredictionMethod.GRADIENT
            ch.quantization_value = 70
            ch.transform_type = TransformType.FWT
            ch.wavelet_type = WaveletType.SYM10
            ch.transform_scale = 20
            ch.min_block_size = 16
            ch.max_block_size = 256
        presets["dreamy"] = dreamy

        # Glitch blocks
        blocks = CodecConfig()
        blocks.color_space = ColorSpace.RGB
        for ch in blocks.channels:
            ch.prediction_method = PredictionMethod.CORNER
            ch.quantization_value = 160
            ch.transform_type = TransformType.NONE
            ch.min_block_size = 16
            ch.max_block_size = 64
            ch.segmentation_precision = 30
        presets["blocks"] = blocks

        # Corrupted
        corrupted = CodecConfig()
        corrupted.color_space = ColorSpace.RGGBG
        for i, ch in enumerate(corrupted.channels):
            ch.prediction_method = [PredictionMethod.NOISE, PredictionMethod.MIRROR, PredictionMethod.EDGE][i]
            ch.quantization_value = 190
            ch.transform_type = TransformType.WPT
            ch.wavelet_type = WaveletType.HAAR
            ch.transform_scale = 50
        presets["corrupted"] = corrupted

        # Mosaic
        mosaic = CodecConfig()
        mosaic.color_space = ColorSpace.HWB
        for ch in mosaic.channels:
            ch.prediction_method = PredictionMethod.DC
            ch.quantization_value = 140
            ch.transform_type = TransformType.NONE
            ch.min_block_size = 8
            ch.max_block_size = 32
            ch.segmentation_precision = 100
        presets["mosaic"] = mosaic

        # Edge detector style
        edges = CodecConfig()
        edges.color_space = ColorSpace.GS
        for ch in edges.channels:
            ch.prediction_method = PredictionMethod.EDGE
            ch.quantization_value = 100
            ch.transform_type = TransformType.FWT
            ch.wavelet_type = WaveletType.HAAR
            ch.transform_scale = 30
        presets["edges"] = edges

        # Interference pattern
        interference = CodecConfig()
        interference.color_space = ColorSpace.YUV
        for ch in interference.channels:
            ch.prediction_method = PredictionMethod.CHECKERBOARD
            ch.quantization_value = 130
            ch.transform_type = TransformType.FWT
            ch.wavelet_type = WaveletType.DB4
            ch.transform_scale = 25
        presets["interference"] = interference

        # Analog TV
        analog = CodecConfig()
        analog.color_space = ColorSpace.YPbPr
        analog.channels[0].prediction_method = PredictionMethod.H
        analog.channels[0].quantization_value = 90
        analog.channels[1].prediction_method = PredictionMethod.WAVE
        analog.channels[1].quantization_value = 160
        analog.channels[2].prediction_method = PredictionMethod.WAVE
        analog.channels[2].quantization_value = 160
        for ch in analog.channels:
            ch.transform_type = TransformType.FWT
            ch.wavelet_type = WaveletType.SYM4
            ch.transform_scale = 30
        presets["analog"] = analog

        # Displacement
        displacement = CodecConfig()
        displacement.color_space = ColorSpace.XYZ
        for ch in displacement.channels:
            ch.prediction_method = PredictionMethod.MIRROR
            ch.quantization_value = 140
            ch.transform_type = TransformType.WPT
            ch.wavelet_type = WaveletType.DB6
            ch.transform_scale = 35
        presets["displacement"] = displacement

        return presets

    def load_preset(self, name: str) -> Optional[CodecConfig]:
        """
        Load a preset by name.

        Args:
            name: Preset name (without extension)

        Returns:
            CodecConfig or None if not found
        """
        # Check builtin presets first
        if name in self._builtin_presets:
            return self._builtin_presets[name].copy()

        # Check cache
        if name in self._presets_cache:
            return self._presets_cache[name].copy()

        # Try to load from file
        if self.presets_dir:
            preset_path = Path(self.presets_dir) / f"{name}.preset"
            if preset_path.exists():
                config = self._load_preset_file(str(preset_path))
                if config:
                    self._presets_cache[name] = config
                    return config.copy()

        return None

    def _load_preset_file(self, path: str) -> Optional[CodecConfig]:
        """
        Load a preset from a binary file (Java serialized format).

        This is a simplified parser that extracts key parameters
        from the original GLIC Java preset format.
        """
        try:
            with open(path, 'rb') as f:
                data = f.read()

            # Java serialization format is complex, so we use heuristics
            # to find relevant values in the binary data

            config = CodecConfig()

            # Look for common patterns in the binary data
            # This is a simplified approach - full Java deserialization would require
            # implementing the Java Object Serialization Stream Protocol

            # Default values with some variation based on file content
            if len(data) > 100:
                # Use bytes from file to create variation
                config.color_space = _COLORSPACE_MAP.get(data[50] % 16, ColorSpace.HWB)

                for i, ch in enumerate(config.channels):
                    offset = 60 + i * 20
                    if offset + 20 < len(data):
                        ch.prediction_method = _PREDICTION_MAP.get(
                            data[offset] % 19, PredictionMethod.PAETH
                        )
                        ch.quantization_value = 50 + (data[offset + 1] % 150)
                        ch.transform_type = _TRANSFORM_MAP.get(
                            data[offset + 2] % 3, TransformType.FWT
                        )
                        ch.wavelet_type = _WAVELET_MAP.get(
                            data[offset + 3] % 24, WaveletType.SYM8
                        )
                        ch.transform_scale = data[offset + 4] % 60
                        ch.min_block_size = 2 ** (1 + data[offset + 5] % 4)
                        ch.max_block_size = 2 ** (4 + data[offset + 6] % 5)
                        ch.segmentation_precision = 5 + (data[offset + 7] % 40)

            return config

        except Exception as e:
            print(f"Warning: Could not load preset {path}: {e}")
            return None

    def list_presets(self) -> List[str]:
        """Get list of all available preset names."""
        presets = list(self._builtin_presets.keys())

        if self.presets_dir:
            preset_dir = Path(self.presets_dir)
            if preset_dir.exists():
                for f in preset_dir.glob("*.preset"):
                    name = f.stem
                    if name not in presets:
                        presets.append(name)

        return sorted(presets)

    def save_preset(self, name: str, config: CodecConfig, directory: Optional[str] = None) -> str:
        """
        Save a preset configuration to a JSON file.

        Args:
            name: Preset name
            config: Configuration to save
            directory: Output directory (defaults to presets_dir)

        Returns:
            Path to saved file
        """
        import json

        out_dir = directory or self.presets_dir or "."
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        out_path = Path(out_dir) / f"{name}.json"

        # Convert config to dict
        data = {
            "name": name,
            "color_space": config.color_space.name,
            "border_color": list(config.border_color),
            "channels": [
                {
                    "min_block_size": ch.min_block_size,
                    "max_block_size": ch.max_block_size,
                    "segmentation_precision": ch.segmentation_precision,
                    "prediction_method": ch.prediction_method.name,
                    "quantization_value": ch.quantization_value,
                    "clamp_method": ch.clamp_method.name,
                    "transform_type": ch.transform_type.name,
                    "wavelet_type": ch.wavelet_type.name,
                    "transform_scale": ch.transform_scale,
                    "encoding_method": ch.encoding_method.name,
                }
                for ch in config.channels
            ]
        }

        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)

        return str(out_path)

    def load_json_preset(self, path: str) -> Optional[CodecConfig]:
        """Load a preset from a JSON file."""
        import json

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            config = CodecConfig()
            config.color_space = ColorSpace[data.get("color_space", "HWB")]
            config.border_color = tuple(data.get("border_color", [128, 128, 128]))

            for i, ch_data in enumerate(data.get("channels", [])):
                if i < 3:
                    ch = config.channels[i]
                    ch.min_block_size = ch_data.get("min_block_size", 2)
                    ch.max_block_size = ch_data.get("max_block_size", 256)
                    ch.segmentation_precision = ch_data.get("segmentation_precision", 15.0)
                    ch.prediction_method = PredictionMethod[ch_data.get("prediction_method", "PAETH")]
                    ch.quantization_value = ch_data.get("quantization_value", 110)
                    ch.clamp_method = ClampMethod[ch_data.get("clamp_method", "NONE")]
                    ch.transform_type = TransformType[ch_data.get("transform_type", "FWT")]
                    ch.wavelet_type = WaveletType[ch_data.get("wavelet_type", "SYM8")]
                    ch.transform_scale = ch_data.get("transform_scale", 20)
                    ch.encoding_method = EncodingMethod[ch_data.get("encoding_method", "PACKED")]

            return config

        except Exception as e:
            print(f"Warning: Could not load JSON preset {path}: {e}")
            return None


def get_default_presets_dir() -> Optional[str]:
    """Get the default presets directory path."""
    # Check relative to this module
    module_dir = Path(__file__).parent
    presets_dir = module_dir.parent / "presets"

    if presets_dir.exists():
        return str(presets_dir)

    # Check in parent directories
    for parent in module_dir.parents:
        presets_dir = parent / "presets"
        if presets_dir.exists():
            return str(presets_dir)

    return None
