"""
Main GLIC codec class.
Orchestrates encoding and decoding of images with glitch effects.
"""

import struct
from typing import List, Tuple, Optional
import numpy as np

from .config import (
    CodecConfig, ChannelConfig, ColorSpace, PredictionMethod,
    EncodingMethod, TransformType, WaveletType, EffectConfig,
    GLIC_MAGIC, GLIC_VERSION
)
from .planes import Planes, load_image, save_image
from .segment import Segment, segment_plane, sort_segments_raster
from .prediction import apply_prediction, apply_inverse_prediction, get_predictor
from .wavelet import apply_transform, apply_inverse_transform
from .encoding import encode_data, decode_data, quantize, dequantize
from .effects import apply_effects
from .bitio import BitWriter, BitReader


class GlicCodec:
    """
    GLIC (GLitch Image Codec) encoder/decoder.

    A codec that intentionally creates artistic glitch effects
    while encoding and decoding images.
    """

    def __init__(self, config: Optional[CodecConfig] = None):
        """
        Initialize the codec with optional configuration.

        Args:
            config: Codec configuration (defaults to standard config)
        """
        self.config = config or CodecConfig()

    def encode(self, input_path: str, output_path: str) -> None:
        """
        Encode an image to GLIC format.

        Args:
            input_path: Path to input image (PNG, JPG, BMP)
            output_path: Path for output GLIC file
        """
        # Load image
        image = load_image(input_path)

        # Create planes from image
        planes = Planes.from_image(
            image,
            self.config.color_space,
            self.config.border_color
        )

        # Encode each channel
        encoded_channels = []
        channel_segments = []

        for ch in range(3):
            plane = planes.get_plane(ch)
            ch_config = self.config.channels[ch]

            # Segment the plane
            segments = segment_plane(
                plane,
                ch_config.min_block_size,
                ch_config.max_block_size,
                ch_config.segmentation_precision,
                ch_config.prediction_method
            )
            segments = sort_segments_raster(segments)
            channel_segments.append(segments)

            # Process each segment
            encoded_data = []
            for seg in segments:
                # Extract block
                block = plane[seg.y:seg.y + seg.height, seg.x:seg.x + seg.width].copy()

                # Apply prediction to get residuals
                residuals = apply_prediction(
                    plane, seg.x, seg.y, seg.width, seg.height,
                    planes.get_ref_value(ch), ch_config.prediction_method
                )

                # Apply wavelet transform
                if ch_config.transform_type != TransformType.NONE:
                    residuals = apply_transform(
                        residuals.astype(np.float64),
                        ch_config.transform_type,
                        ch_config.wavelet_type,
                        ch_config.transform_scale
                    )

                # Quantize
                residuals = (residuals / max(1, ch_config.quantization_value / 100)).astype(np.int32)

                encoded_data.append(residuals)

            encoded_channels.append(encoded_data)

        # Write to file
        self._write_glic_file(
            output_path, planes, channel_segments, encoded_channels
        )

    def decode(self, input_path: str, output_path: str) -> None:
        """
        Decode a GLIC file to an image.

        Args:
            input_path: Path to GLIC file
            output_path: Path for output image (PNG)
        """
        # Read GLIC file
        planes, channel_segments, encoded_channels = self._read_glic_file(input_path)

        # Decode each channel
        for ch in range(3):
            plane = planes.get_plane(ch)
            ch_config = self.config.channels[ch]
            segments = channel_segments[ch]
            encoded_data = encoded_channels[ch]

            # Clear plane
            plane.fill(planes.get_ref_value(ch))

            # Process each segment
            for i, seg in enumerate(segments):
                residuals = encoded_data[i].astype(np.float64)

                # Dequantize
                residuals = residuals * (ch_config.quantization_value / 100)

                # Inverse wavelet transform
                if ch_config.transform_type != TransformType.NONE:
                    residuals = apply_inverse_transform(
                        residuals,
                        ch_config.transform_type,
                        ch_config.wavelet_type,
                        (seg.height, seg.width)
                    )

                # Apply inverse prediction
                apply_inverse_prediction(
                    residuals.astype(np.int32),
                    plane, seg.x, seg.y,
                    planes.get_ref_value(ch),
                    ch_config.prediction_method
                )

            planes.set_plane(ch, plane)

        # Convert to image
        image = planes.to_image(self.config.color_space)

        # Apply post-processing effects
        if self.config.effects:
            image = apply_effects(image, self.config.effects)

        # Save output
        save_image(image, output_path)

    def encode_decode(self, input_path: str, output_path: str) -> None:
        """
        Encode and immediately decode an image (for glitch effect).

        This is a convenience method that applies the glitch effect
        without creating an intermediate GLIC file.
        """
        # Load image
        image = load_image(input_path)

        # Create planes
        planes = Planes.from_image(
            image,
            self.config.color_space,
            self.config.border_color
        )

        # Process each channel
        for ch in range(3):
            plane = planes.get_plane(ch)
            ch_config = self.config.channels[ch]

            # Create working copy
            output_plane = np.full_like(plane, planes.get_ref_value(ch))

            # Segment
            segments = segment_plane(
                plane,
                ch_config.min_block_size,
                ch_config.max_block_size,
                ch_config.segmentation_precision,
                ch_config.prediction_method
            )
            segments = sort_segments_raster(segments)

            # Process each segment
            for seg in segments:
                # Get residuals
                residuals = apply_prediction(
                    plane, seg.x, seg.y, seg.width, seg.height,
                    planes.get_ref_value(ch), ch_config.prediction_method
                )

                # Apply wavelet transform
                if ch_config.transform_type != TransformType.NONE:
                    residuals = apply_transform(
                        residuals.astype(np.float64),
                        ch_config.transform_type,
                        ch_config.wavelet_type,
                        ch_config.transform_scale
                    )

                # Quantize and dequantize (lossy step)
                quant_factor = ch_config.quantization_value / 100
                residuals = (residuals / max(1, quant_factor)).astype(np.int32)
                residuals = (residuals * quant_factor).astype(np.float64)

                # Inverse wavelet
                if ch_config.transform_type != TransformType.NONE:
                    residuals = apply_inverse_transform(
                        residuals,
                        ch_config.transform_type,
                        ch_config.wavelet_type,
                        (seg.height, seg.width)
                    )

                # Inverse prediction
                apply_inverse_prediction(
                    residuals.astype(np.int32),
                    output_plane, seg.x, seg.y,
                    planes.get_ref_value(ch),
                    ch_config.prediction_method
                )

            planes.set_plane(ch, output_plane)

        # Convert to image
        result = planes.to_image(self.config.color_space)

        # Apply effects
        if self.config.effects:
            result = apply_effects(result, self.config.effects)

        # Save
        save_image(result, output_path)

    def _write_glic_file(self, path: str, planes: Planes,
                         channel_segments: List[List[Segment]],
                         encoded_channels: List[List[np.ndarray]]) -> None:
        """Write encoded data to GLIC file."""
        writer = BitWriter()

        # Magic number and version
        writer.write_uint32(GLIC_MAGIC)
        writer.write_uint16(GLIC_VERSION)

        # Image dimensions
        orig_w, orig_h = planes.get_original_size()
        pad_w, pad_h = planes.get_padded_size()
        writer.write_uint16(orig_w)
        writer.write_uint16(orig_h)
        writer.write_uint16(pad_w)
        writer.write_uint16(pad_h)

        # Color space
        writer.write_byte(self.config.color_space.value)

        # Border color
        writer.write_byte(self.config.border_color[0])
        writer.write_byte(self.config.border_color[1])
        writer.write_byte(self.config.border_color[2])

        # Channel configurations
        for ch_config in self.config.channels:
            writer.write_byte(ch_config.min_block_size)
            writer.write_uint16(ch_config.max_block_size)
            writer.write_float(ch_config.segmentation_precision)
            writer.write_byte(ch_config.prediction_method.value)
            writer.write_byte(ch_config.quantization_value)
            writer.write_byte(ch_config.clamp_method.value)
            writer.write_byte(ch_config.transform_type.value)
            writer.write_byte(ch_config.wavelet_type.value)
            writer.write_byte(ch_config.transform_scale)
            writer.write_byte(ch_config.encoding_method.value)

        # Write channel data
        for ch in range(3):
            segments = channel_segments[ch]
            encoded_data = encoded_channels[ch]
            ch_config = self.config.channels[ch]

            # Number of segments
            writer.write_uint32(len(segments))

            # Segment info
            for seg in segments:
                writer.write_uint16(seg.x)
                writer.write_uint16(seg.y)
                writer.write_uint16(seg.width)
                writer.write_uint16(seg.height)

            # Encoded residuals
            for residuals in encoded_data:
                data_bytes = encode_data(residuals, ch_config.encoding_method)
                writer.write_uint32(len(data_bytes))
                for b in data_bytes:
                    writer.write_byte(b)

        # Write to file
        with open(path, 'wb') as f:
            f.write(writer.get_bytes())

    def _read_glic_file(self, path: str) -> Tuple[Planes, List[List[Segment]], List[List[np.ndarray]]]:
        """Read GLIC file and return decoded data."""
        with open(path, 'rb') as f:
            data = f.read()

        reader = BitReader(data)

        # Verify magic number
        magic = reader.read_uint32()
        if magic != GLIC_MAGIC:
            raise ValueError(f"Invalid GLIC file: wrong magic number {hex(magic)}")

        version = reader.read_uint16()
        if version > GLIC_VERSION:
            raise ValueError(f"Unsupported GLIC version: {version}")

        # Read dimensions
        orig_w = reader.read_uint16()
        orig_h = reader.read_uint16()
        pad_w = reader.read_uint16()
        pad_h = reader.read_uint16()

        # Color space
        cs_val = reader.read_byte()
        self.config.color_space = ColorSpace(cs_val)

        # Border color
        self.config.border_color = (
            reader.read_byte(),
            reader.read_byte(),
            reader.read_byte()
        )

        # Create planes
        planes = Planes(orig_w, orig_h, self.config.border_color)

        # Read channel configurations
        for i in range(3):
            ch = self.config.channels[i]
            ch.min_block_size = reader.read_byte()
            ch.max_block_size = reader.read_uint16()
            ch.segmentation_precision = reader.read_float()
            ch.prediction_method = PredictionMethod(reader.read_byte())
            ch.quantization_value = reader.read_byte()
            ch.clamp_method = ClampMethod(reader.read_byte())
            ch.transform_type = TransformType(reader.read_byte())
            ch.wavelet_type = WaveletType(reader.read_byte())
            ch.transform_scale = reader.read_byte()
            ch.encoding_method = EncodingMethod(reader.read_byte())

        # Read channel data
        channel_segments = []
        encoded_channels = []

        for ch in range(3):
            ch_config = self.config.channels[ch]

            # Read segments
            num_segments = reader.read_uint32()
            segments = []
            for _ in range(num_segments):
                seg = Segment(
                    x=reader.read_uint16(),
                    y=reader.read_uint16(),
                    width=reader.read_uint16(),
                    height=reader.read_uint16(),
                    prediction_method=ch_config.prediction_method
                )
                segments.append(seg)

            channel_segments.append(segments)

            # Read encoded data
            encoded_data = []
            for _ in range(num_segments):
                data_len = reader.read_uint32()
                data_bytes = bytes([reader.read_byte() for _ in range(data_len)])
                residuals = decode_data(data_bytes, ch_config.encoding_method)
                encoded_data.append(residuals)

            encoded_channels.append(encoded_data)

        return planes, channel_segments, encoded_channels

    def add_effect(self, effect: EffectConfig) -> None:
        """Add a post-processing effect."""
        self.config.effects.append(effect)

    def clear_effects(self) -> None:
        """Remove all post-processing effects."""
        self.config.effects.clear()

    def set_config(self, config: CodecConfig) -> None:
        """Set the codec configuration."""
        self.config = config

    def get_config(self) -> CodecConfig:
        """Get the current configuration."""
        return self.config


# Import ClampMethod for file reading
from .config import ClampMethod
