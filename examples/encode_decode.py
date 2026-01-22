#!/usr/bin/env python3
"""
GLIC Encode/Decode Examples

This script demonstrates how to use the GLIC format (.glic files) for:
- Encoding images to GLIC format
- Decoding GLIC files back to images
- Comparing file sizes
- Re-processing GLIC files with different settings
- Batch encode/decode operations

The GLIC format stores the encoded image data along with all configuration
parameters, allowing you to:
1. Store glitched images in a compact format
2. Re-decode with different post-processing effects
3. Share configurations embedded in the file
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from glic.codec import GlicCodec
from glic.config import (
    CodecConfig, ChannelConfig, ColorSpace, PredictionMethod,
    TransformType, WaveletType, EffectType
)
from glic.preset_loader import PresetLoader
from glic.effects import create_effect_config


def example_basic_encode_decode(input_path: str, output_dir: str):
    """
    Example 1: Basic encode and decode operations.

    Demonstrates the fundamental workflow:
    - Encode an image to .glic format
    - Decode the .glic file back to an image
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Encode/Decode")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a codec with default settings
    codec = GlicCodec()

    # Encode image to GLIC format
    glic_file = str(output_path / "encoded.glic")
    print(f"\nEncoding: {input_path}")
    print(f"      to: {glic_file}")
    codec.encode(input_path, glic_file)

    # Decode GLIC file back to image
    decoded_file = str(output_path / "decoded.png")
    print(f"\nDecoding: {glic_file}")
    print(f"      to: {decoded_file}")
    codec.decode(glic_file, decoded_file)

    # Compare file sizes
    original_size = os.path.getsize(input_path)
    glic_size = os.path.getsize(glic_file)
    decoded_size = os.path.getsize(decoded_file)

    print(f"\nFile sizes:")
    print(f"  Original: {original_size:,} bytes")
    print(f"  GLIC:     {glic_size:,} bytes ({glic_size/original_size*100:.1f}%)")
    print(f"  Decoded:  {decoded_size:,} bytes")

    return glic_file


def example_encode_with_preset(input_path: str, output_dir: str):
    """
    Example 2: Encode with different presets.

    Shows how presets affect the encoded GLIC file.
    """
    print("\n" + "=" * 60)
    print("Example 2: Encode with Different Presets")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loader = PresetLoader()
    presets_to_try = ["default", "heavy_glitch", "minimal", "vhs"]

    results = []

    for preset_name in presets_to_try:
        config = loader.load_preset(preset_name)
        if config is None:
            print(f"  Preset '{preset_name}' not found, skipping")
            continue

        codec = GlicCodec(config)

        glic_file = str(output_path / f"preset_{preset_name}.glic")
        decoded_file = str(output_path / f"preset_{preset_name}.png")

        # Encode and decode
        codec.encode(input_path, glic_file)
        codec.decode(glic_file, decoded_file)

        glic_size = os.path.getsize(glic_file)
        results.append((preset_name, glic_size, glic_file, decoded_file))

        print(f"  {preset_name}: {glic_size:,} bytes -> {decoded_file}")

    return results


def example_redecode_with_effects(glic_file: str, output_dir: str):
    """
    Example 3: Re-decode a GLIC file with different effects.

    A key advantage of the GLIC format: you can decode the same file
    multiple times with different post-processing effects.
    """
    print("\n" + "=" * 60)
    print("Example 3: Re-decode with Different Effects")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Effect combinations to try
    effect_combinations = [
        ("no_effects", []),
        ("scanline", [EffectType.SCANLINE]),
        ("chromatic", [EffectType.CHROMATIC]),
        ("pixelate", [EffectType.PIXELATE]),
        ("combined", [EffectType.SCANLINE, EffectType.CHROMATIC]),
        ("retro", [EffectType.SCANLINE, EffectType.DITHER, EffectType.POSTERIZE]),
    ]

    print(f"\nSource GLIC: {glic_file}")
    print("\nDecoding with different effects:")

    for name, effects in effect_combinations:
        # Create codec with effects
        config = CodecConfig()
        for effect_type in effects:
            config.effects.append(create_effect_config(effect_type, intensity=50))

        codec = GlicCodec(config)

        decoded_file = str(output_path / f"effects_{name}.png")
        codec.decode(glic_file, decoded_file)

        effect_names = [e.name.lower() for e in effects] if effects else ["none"]
        print(f"  {name}: {', '.join(effect_names)} -> {decoded_file}")


def example_custom_config_encode(input_path: str, output_dir: str):
    """
    Example 4: Encode with custom configuration.

    Demonstrates fine-grained control over encoding parameters.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration Encoding")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create custom configuration
    config = CodecConfig(
        color_space=ColorSpace.YUV,
        channels=[
            ChannelConfig(
                prediction_method=PredictionMethod.SPIRAL,
                quantization_value=120,
                min_block_size=4,
                max_block_size=64,
                transform_type=TransformType.FWT,
                wavelet_type=WaveletType.HAAR,
                transform_scale=2,
            ),
            ChannelConfig(
                prediction_method=PredictionMethod.WAVE,
                quantization_value=150,
                min_block_size=8,
                max_block_size=128,
            ),
            ChannelConfig(
                prediction_method=PredictionMethod.RADIAL,
                quantization_value=180,
                min_block_size=8,
                max_block_size=128,
            ),
        ]
    )

    codec = GlicCodec(config)

    glic_file = str(output_path / "custom_config.glic")
    decoded_file = str(output_path / "custom_config.png")

    print(f"\nCustom configuration:")
    print(f"  Color space: {config.color_space.name}")
    print(f"  Channel 0: {config.channels[0].prediction_method.name}, "
          f"quant={config.channels[0].quantization_value}")
    print(f"  Channel 1: {config.channels[1].prediction_method.name}, "
          f"quant={config.channels[1].quantization_value}")
    print(f"  Channel 2: {config.channels[2].prediction_method.name}, "
          f"quant={config.channels[2].quantization_value}")

    # Encode and decode
    codec.encode(input_path, glic_file)
    codec.decode(glic_file, decoded_file)

    print(f"\nEncoded to: {glic_file}")
    print(f"Decoded to: {decoded_file}")
    print(f"GLIC size:  {os.path.getsize(glic_file):,} bytes")

    return glic_file


def example_direct_glitch(input_path: str, output_dir: str):
    """
    Example 5: Direct glitch (encode + decode in one step).

    Compare this with the two-step encode/decode approach.
    """
    print("\n" + "=" * 60)
    print("Example 5: Direct Glitch vs Encode/Decode")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loader = PresetLoader()
    config = loader.load_preset("heavy_glitch") or CodecConfig()

    codec = GlicCodec(config)

    # Method 1: Direct glitch (no intermediate file)
    direct_output = str(output_path / "direct_glitch.png")
    print(f"\nMethod 1: Direct glitch (encode_decode)")
    codec.encode_decode(input_path, direct_output)
    print(f"  Output: {direct_output}")

    # Method 2: Two-step encode then decode
    glic_file = str(output_path / "two_step.glic")
    decode_output = str(output_path / "two_step_decoded.png")
    print(f"\nMethod 2: Two-step (encode -> decode)")
    codec.encode(input_path, glic_file)
    codec.decode(glic_file, decode_output)
    print(f"  GLIC file: {glic_file}")
    print(f"  Output: {decode_output}")

    print(f"\nNote: Both methods produce equivalent results.")
    print(f"Use encode/decode when you want to:")
    print(f"  - Store the intermediate GLIC file")
    print(f"  - Re-decode with different effects later")
    print(f"  - Share the GLIC file with others")


def example_batch_encode(input_dir: str, output_dir: str):
    """
    Example 6: Batch encode multiple images.

    Useful for processing entire directories.
    """
    print("\n" + "=" * 60)
    print("Example 6: Batch Encoding")
    print("=" * 60)

    input_path = Path(input_dir)
    output_path = Path(output_dir) / "batch"
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    input_files = []
    for ext in extensions:
        input_files.extend(input_path.glob(ext))

    if not input_files:
        print(f"\nNo images found in {input_dir}")
        print("Skipping batch encoding example.")
        return

    loader = PresetLoader()
    config = loader.load_preset("default") or CodecConfig()
    codec = GlicCodec(config)

    print(f"\nFound {len(input_files)} images to process")

    for i, img_file in enumerate(input_files[:5]):  # Limit to 5 for demo
        glic_file = output_path / f"{img_file.stem}.glic"
        decoded_file = output_path / f"{img_file.stem}_glitched.png"

        codec.encode(str(img_file), str(glic_file))
        codec.decode(str(glic_file), str(decoded_file))

        print(f"  [{i+1}/{min(len(input_files), 5)}] {img_file.name} -> {glic_file.name}")


def print_usage():
    """Print usage information."""
    print("GLIC Encode/Decode Examples")
    print("=" * 40)
    print()
    print("Usage: python encode_decode.py <image_path> [output_dir]")
    print()
    print("Arguments:")
    print("  image_path  Path to input image (PNG, JPG, BMP)")
    print("  output_dir  Output directory (default: ./encode_decode_output)")
    print()
    print("Examples:")
    print("  python encode_decode.py photo.png")
    print("  python encode_decode.py photo.png ./my_output")
    print()
    print("This script demonstrates:")
    print("  1. Basic encode/decode operations")
    print("  2. Encoding with different presets")
    print("  3. Re-decoding with different effects")
    print("  4. Custom configuration encoding")
    print("  5. Direct glitch vs encode/decode comparison")
    print("  6. Batch encoding (if input is a directory)")


def main():
    """Run all encode/decode examples."""
    if len(sys.argv) < 2:
        print_usage()
        return

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./encode_decode_output"

    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        return

    print("GLIC Encode/Decode Examples")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")

    # Run examples
    if os.path.isfile(input_path):
        # Example 1: Basic encode/decode
        glic_file = example_basic_encode_decode(input_path, output_dir)

        # Example 2: Encode with presets
        example_encode_with_preset(input_path, output_dir)

        # Example 3: Re-decode with effects
        example_redecode_with_effects(glic_file, output_dir)

        # Example 4: Custom config
        example_custom_config_encode(input_path, output_dir)

        # Example 5: Direct glitch comparison
        example_direct_glitch(input_path, output_dir)

    elif os.path.isdir(input_path):
        # Example 6: Batch encoding
        example_batch_encode(input_path, output_dir)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print(f"Check output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
