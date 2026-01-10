"""
GLIC Command Line Interface.

Usage:
    python -m glic encode input.png output.glic [options]
    python -m glic decode input.glic output.png [options]
    python -m glic glitch input.png output.png [options]
    python -m glic --list-presets
"""

import argparse
import sys
from pathlib import Path

from .codec import GlicCodec
from .config import (
    CodecConfig, ChannelConfig, ColorSpace, PredictionMethod,
    EncodingMethod, TransformType, WaveletType, EffectType, EffectConfig
)
from .preset_loader import PresetLoader, get_default_presets_dir
from .effects import create_effect_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="glic",
        description="GLIC - GLitch Image Codec: Create artistic glitch effects on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Apply glitch effect directly
    python -m glic glitch input.png output.png

    # Use a preset
    python -m glic glitch input.png output.png --preset heavy_glitch

    # Encode to GLIC format
    python -m glic encode input.png output.glic

    # Decode GLIC file
    python -m glic decode input.glic output.png

    # Custom parameters
    python -m glic glitch input.png output.png \\
        --colorspace YUV --prediction SPIRAL --quantization 150

    # Add post-processing effects
    python -m glic glitch input.png output.png \\
        --effect scanline --effect chromatic

    # List available presets
    python -m glic --list-presets
        """
    )

    parser.add_argument("--version", action="version", version="GLIC 1.0.0")

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available presets"
    )

    parser.add_argument(
        "--list-colorspaces",
        action="store_true",
        help="List all available color spaces"
    )

    parser.add_argument(
        "--list-predictions",
        action="store_true",
        help="List all available prediction methods"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode image to GLIC format")
    encode_parser.add_argument("input", help="Input image file (PNG, JPG, BMP)")
    encode_parser.add_argument("output", help="Output GLIC file")
    add_common_args(encode_parser)

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode GLIC file to image")
    decode_parser.add_argument("input", help="Input GLIC file")
    decode_parser.add_argument("output", help="Output image file (PNG)")
    add_common_args(decode_parser)

    # Glitch command (encode + decode in one step)
    glitch_parser = subparsers.add_parser("glitch", help="Apply glitch effect directly")
    glitch_parser.add_argument("input", help="Input image file (PNG, JPG, BMP)")
    glitch_parser.add_argument("output", help="Output image file (PNG)")
    add_common_args(glitch_parser)

    return parser.parse_args()


def add_common_args(parser):
    """Add common arguments to a subparser."""
    parser.add_argument(
        "--preset", "-p",
        help="Use a preset configuration"
    )

    parser.add_argument(
        "--presets-dir",
        help="Directory containing preset files"
    )

    parser.add_argument(
        "--colorspace", "-c",
        choices=[cs.name for cs in ColorSpace],
        help="Color space for processing"
    )

    parser.add_argument(
        "--prediction", "-P",
        choices=[pm.name for pm in PredictionMethod],
        help="Prediction method"
    )

    parser.add_argument(
        "--quantization", "-q",
        type=int,
        help="Quantization value (0-255)"
    )

    parser.add_argument(
        "--min-block",
        type=int,
        help="Minimum block size"
    )

    parser.add_argument(
        "--max-block",
        type=int,
        help="Maximum block size"
    )

    parser.add_argument(
        "--precision",
        type=float,
        help="Segmentation precision"
    )

    parser.add_argument(
        "--transform",
        choices=[tt.name for tt in TransformType],
        help="Wavelet transform type"
    )

    parser.add_argument(
        "--wavelet",
        choices=[wt.name for wt in WaveletType],
        help="Wavelet type"
    )

    parser.add_argument(
        "--transform-scale",
        type=int,
        help="Transform scale (compression level)"
    )

    parser.add_argument(
        "--encoding",
        choices=[em.name for em in EncodingMethod],
        help="Encoding method"
    )

    parser.add_argument(
        "--effect", "-e",
        action="append",
        choices=[et.name.lower() for et in EffectType],
        help="Add post-processing effect (can be used multiple times)"
    )

    parser.add_argument(
        "--effect-intensity",
        type=int,
        default=50,
        help="Effect intensity (0-100)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )


def list_items(title: str, items: list):
    """Print a list of items."""
    print(f"\n{title}:")
    print("-" * len(title))
    for item in items:
        print(f"  {item}")
    print()


def build_config(args, preset_loader: PresetLoader) -> CodecConfig:
    """Build codec configuration from arguments."""
    # Start with preset or default
    if args.preset:
        config = preset_loader.load_preset(args.preset)
        if config is None:
            print(f"Warning: Preset '{args.preset}' not found, using default")
            config = CodecConfig()
    else:
        config = CodecConfig()

    # Apply command-line overrides
    if args.colorspace:
        config.color_space = ColorSpace[args.colorspace]

    for ch in config.channels:
        if args.prediction:
            ch.prediction_method = PredictionMethod[args.prediction]
        if args.quantization is not None:
            ch.quantization_value = args.quantization
        if args.min_block is not None:
            ch.min_block_size = args.min_block
        if args.max_block is not None:
            ch.max_block_size = args.max_block
        if args.precision is not None:
            ch.segmentation_precision = args.precision
        if args.transform:
            ch.transform_type = TransformType[args.transform]
        if args.wavelet:
            ch.wavelet_type = WaveletType[args.wavelet]
        if args.transform_scale is not None:
            ch.transform_scale = args.transform_scale
        if args.encoding:
            ch.encoding_method = EncodingMethod[args.encoding]

    # Add effects
    if args.effect:
        for effect_name in args.effect:
            effect_type = EffectType[effect_name.upper()]
            effect = create_effect_config(
                effect_type,
                intensity=args.effect_intensity
            )
            config.effects.append(effect)

    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Handle list commands
    if args.list_presets:
        presets_dir = get_default_presets_dir()
        loader = PresetLoader(presets_dir)
        presets = loader.list_presets()
        list_items("Available Presets", presets)
        return 0

    if args.list_colorspaces:
        list_items("Available Color Spaces", [cs.name for cs in ColorSpace])
        return 0

    if args.list_predictions:
        list_items("Available Prediction Methods", [pm.name for pm in PredictionMethod])
        return 0

    # Check for command
    if not args.command:
        print("Error: No command specified. Use --help for usage information.")
        return 1

    # Initialize preset loader
    presets_dir = getattr(args, 'presets_dir', None) or get_default_presets_dir()
    preset_loader = PresetLoader(presets_dir)

    # Build configuration
    config = build_config(args, preset_loader)

    # Create codec
    codec = GlicCodec(config)

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Execute command
    if args.verbose:
        print(f"Command: {args.command}")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Color space: {config.color_space.name}")
        print(f"Prediction: {config.channels[0].prediction_method.name}")
        print(f"Quantization: {config.channels[0].quantization_value}")

    try:
        if args.command == "encode":
            if args.verbose:
                print("Encoding...")
            codec.encode(args.input, args.output)
            print(f"Encoded: {args.output}")

        elif args.command == "decode":
            if args.verbose:
                print("Decoding...")
            codec.decode(args.input, args.output)
            print(f"Decoded: {args.output}")

        elif args.command == "glitch":
            if args.verbose:
                print("Applying glitch effect...")
            codec.encode_decode(args.input, args.output)
            print(f"Glitched: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
