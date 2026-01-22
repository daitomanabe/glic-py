#!/usr/bin/env python3
"""
GLIC GIF Animation Examples

This script demonstrates how to create animated GIFs with glitch effects:
- Glitch intensity transitions (fade in/out)
- Cycling through different presets
- Oscillating parameters for pulsing effects
- Random glitch sequences
- Smooth interpolation between glitch states

Perfect for creating eye-catching glitch art animations!
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from glic.codec import GlicCodec
from glic.config import (
    CodecConfig, ChannelConfig, ColorSpace, PredictionMethod,
    TransformType, EffectType
)
from glic.preset_loader import PresetLoader
from glic.planes import load_image, save_image
from glic.effects import create_effect_config


def create_gif(frames: List[np.ndarray], output_path: str,
               duration: int = 100, loop: int = 0):
    """
    Create an animated GIF from a list of frames.

    Args:
        frames: List of RGB numpy arrays
        output_path: Path for output GIF
        duration: Duration per frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow is required for GIF creation")
        print("Install with: pip install Pillow")
        return

    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )


def glitch_frame(image: np.ndarray, config: CodecConfig) -> np.ndarray:
    """Apply glitch effect to a single frame using in-memory processing."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")

        save_image(image, input_path)
        codec = GlicCodec(config)
        codec.encode_decode(input_path, output_path)

        return load_image(output_path)


def example_intensity_fade(input_path: str, output_dir: str):
    """
    Example 1: Intensity fade in/out animation.

    Creates a smooth transition from no glitch to heavy glitch and back.
    """
    print("\n" + "=" * 60)
    print("Example 1: Intensity Fade Animation")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    frames = []
    num_frames = 30

    print(f"Generating {num_frames} frames...")

    for i in range(num_frames):
        # Sine wave for smooth fade in/out
        t = i / num_frames
        intensity = (math.sin(t * 2 * math.pi - math.pi/2) + 1) / 2  # 0 to 1 to 0

        # Map intensity to quantization (50-200)
        quant = int(50 + intensity * 150)
        scale = int(intensity * 30)

        config = CodecConfig()
        for ch in config.channels:
            ch.quantization_value = quant
            ch.transform_scale = scale
            ch.prediction_method = PredictionMethod.PAETH

        frame = glitch_frame(image, config)
        frames.append(frame)
        print(f"  Frame {i+1}/{num_frames}: intensity={intensity:.2f}")

    # Create GIF
    gif_path = str(output_path / "intensity_fade.gif")
    create_gif(frames, gif_path, duration=100)
    print(f"\nGIF saved: {gif_path}")

    return gif_path


def example_preset_cycle(input_path: str, output_dir: str):
    """
    Example 2: Cycle through different presets.

    Shows the variety of glitch styles available.
    """
    print("\n" + "=" * 60)
    print("Example 2: Preset Cycle Animation")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    loader = PresetLoader()

    presets = ["default", "heavy_glitch", "vhs", "acid", "dreamy",
               "blocks", "mosaic", "interference", "analog", "chromatic"]

    frames = []
    frames_per_preset = 5

    print(f"Cycling through {len(presets)} presets...")

    for preset_name in presets:
        config = loader.load_preset(preset_name)
        if config is None:
            continue

        # Add multiple frames per preset for longer display
        frame = glitch_frame(image, config)
        for _ in range(frames_per_preset):
            frames.append(frame)

        print(f"  Added: {preset_name}")

    # Create GIF
    gif_path = str(output_path / "preset_cycle.gif")
    create_gif(frames, gif_path, duration=200)
    print(f"\nGIF saved: {gif_path}")

    return gif_path


def example_oscillating_params(input_path: str, output_dir: str):
    """
    Example 3: Oscillating parameters for pulsing effect.

    Creates a hypnotic pulsing glitch animation.
    """
    print("\n" + "=" * 60)
    print("Example 3: Oscillating Parameters Animation")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    frames = []
    num_frames = 40

    print(f"Generating {num_frames} frames with oscillating parameters...")

    predictions = [
        PredictionMethod.SPIRAL,
        PredictionMethod.WAVE,
        PredictionMethod.RADIAL,
        PredictionMethod.CHECKERBOARD
    ]

    for i in range(num_frames):
        t = i / num_frames * 2 * math.pi

        # Oscillate different parameters
        quant = int(100 + 80 * math.sin(t))  # 20-180
        block_size = int(16 + 48 * (math.sin(t * 2) + 1) / 2)  # 16-64
        precision = 10 + 30 * (math.sin(t * 3) + 1) / 2  # 10-40

        # Cycle through predictions
        pred_idx = int(i / num_frames * len(predictions)) % len(predictions)

        config = CodecConfig(color_space=ColorSpace.YUV)
        for ch in config.channels:
            ch.quantization_value = quant
            ch.max_block_size = block_size
            ch.segmentation_precision = precision
            ch.prediction_method = predictions[pred_idx]

        frame = glitch_frame(image, config)
        frames.append(frame)

        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{num_frames}")

    # Create GIF
    gif_path = str(output_path / "oscillating.gif")
    create_gif(frames, gif_path, duration=80)
    print(f"\nGIF saved: {gif_path}")

    return gif_path


def example_random_glitch_sequence(input_path: str, output_dir: str):
    """
    Example 4: Random glitch sequence.

    Creates an unpredictable, chaotic glitch animation.
    """
    print("\n" + "=" * 60)
    print("Example 4: Random Glitch Sequence")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    frames = []
    num_frames = 25

    # Set seed for reproducibility
    np.random.seed(42)

    colorspaces = [ColorSpace.RGB, ColorSpace.YUV, ColorSpace.HSB,
                   ColorSpace.HWB, ColorSpace.OHTA, ColorSpace.LAB]

    predictions = [PredictionMethod.PAETH, PredictionMethod.SPIRAL,
                   PredictionMethod.WAVE, PredictionMethod.RADIAL,
                   PredictionMethod.GRADIENT, PredictionMethod.NOISE]

    print(f"Generating {num_frames} random glitch frames...")

    for i in range(num_frames):
        config = CodecConfig(
            color_space=np.random.choice(colorspaces)
        )

        for ch in config.channels:
            ch.quantization_value = np.random.randint(50, 200)
            ch.prediction_method = np.random.choice(predictions)
            ch.min_block_size = np.random.choice([2, 4, 8])
            ch.max_block_size = np.random.choice([32, 64, 128])
            ch.segmentation_precision = np.random.uniform(5, 50)

            # Random transform
            if np.random.random() > 0.5:
                ch.transform_type = TransformType.FWT
                ch.transform_scale = np.random.randint(1, 4)

        frame = glitch_frame(image, config)
        frames.append(frame)
        print(f"  Frame {i+1}/{num_frames}")

    # Create GIF
    gif_path = str(output_path / "random_sequence.gif")
    create_gif(frames, gif_path, duration=150)
    print(f"\nGIF saved: {gif_path}")

    return gif_path


def example_effect_animation(input_path: str, output_dir: str):
    """
    Example 5: Animated post-processing effects.

    Shows effects being applied with varying intensities.
    """
    print("\n" + "=" * 60)
    print("Example 5: Effect Animation")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    frames = []
    num_frames = 30

    print(f"Generating {num_frames} frames with animated effects...")

    for i in range(num_frames):
        t = i / num_frames

        config = CodecConfig()
        for ch in config.channels:
            ch.quantization_value = 100
            ch.prediction_method = PredictionMethod.SPIRAL

        # Animate effect intensity
        scanline_intensity = int(30 + 50 * (math.sin(t * 2 * math.pi) + 1) / 2)
        chromatic_intensity = int(20 + 60 * (math.cos(t * 2 * math.pi) + 1) / 2)

        config.effects.append(create_effect_config(
            EffectType.SCANLINE, intensity=scanline_intensity
        ))
        config.effects.append(create_effect_config(
            EffectType.CHROMATIC, intensity=chromatic_intensity
        ))

        frame = glitch_frame(image, config)
        frames.append(frame)

        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{num_frames}")

    # Create GIF
    gif_path = str(output_path / "effect_animation.gif")
    create_gif(frames, gif_path, duration=80)
    print(f"\nGIF saved: {gif_path}")

    return gif_path


def example_colorspace_morph(input_path: str, output_dir: str):
    """
    Example 6: Morphing between color spaces.

    Shows how different color spaces produce different glitch aesthetics.
    """
    print("\n" + "=" * 60)
    print("Example 6: Color Space Morph Animation")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    frames = []

    colorspaces = [
        ColorSpace.RGB,
        ColorSpace.YUV,
        ColorSpace.HSB,
        ColorSpace.LAB,
        ColorSpace.OHTA,
        ColorSpace.HWB,
    ]

    frames_per_colorspace = 5

    print(f"Morphing through {len(colorspaces)} color spaces...")

    for cs in colorspaces:
        config = CodecConfig(color_space=cs)
        for ch in config.channels:
            ch.quantization_value = 120
            ch.prediction_method = PredictionMethod.WAVE

        frame = glitch_frame(image, config)

        # Add multiple frames for each colorspace
        for _ in range(frames_per_colorspace):
            frames.append(frame)

        print(f"  Added: {cs.name}")

    # Create GIF
    gif_path = str(output_path / "colorspace_morph.gif")
    create_gif(frames, gif_path, duration=150)
    print(f"\nGIF saved: {gif_path}")

    return gif_path


def example_strobe_effect(input_path: str, output_dir: str):
    """
    Example 7: Strobe/flash effect.

    Alternates between original and heavily glitched versions.
    """
    print("\n" + "=" * 60)
    print("Example 7: Strobe Effect Animation")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    frames = []
    num_cycles = 10

    print(f"Creating strobe effect with {num_cycles} cycles...")

    # Create heavily glitched version
    loader = PresetLoader()
    heavy_config = loader.load_preset("heavy_glitch") or CodecConfig()
    glitched = glitch_frame(image, heavy_config)

    # Alternate between original and glitched
    for i in range(num_cycles):
        # Original (or slightly glitched)
        frames.append(image.copy())
        # Heavily glitched
        frames.append(glitched.copy())

    # Create GIF with fast frame rate for strobe effect
    gif_path = str(output_path / "strobe.gif")
    create_gif(frames, gif_path, duration=50)  # Fast switching
    print(f"\nGIF saved: {gif_path}")

    return gif_path


def print_usage():
    """Print usage information."""
    print("GLIC GIF Animation Examples")
    print("=" * 40)
    print()
    print("Usage: python gif_animation.py <image_path> [output_dir]")
    print()
    print("Arguments:")
    print("  image_path  Path to input image (PNG, JPG, BMP)")
    print("  output_dir  Output directory (default: ./gif_output)")
    print()
    print("This script creates animated GIFs demonstrating:")
    print("  1. Intensity fade in/out")
    print("  2. Preset cycling")
    print("  3. Oscillating parameters")
    print("  4. Random glitch sequences")
    print("  5. Animated effects")
    print("  6. Color space morphing")
    print("  7. Strobe effect")
    print()
    print("Note: Requires Pillow for GIF creation")


def main():
    """Run all GIF animation examples."""
    if len(sys.argv) < 2:
        print_usage()
        return

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./gif_output"

    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        return

    # Check for Pillow
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow is required for GIF creation")
        print("Install with: pip install Pillow")
        return

    print("GLIC GIF Animation Examples")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")

    # Run all examples
    example_intensity_fade(input_path, output_dir)
    example_preset_cycle(input_path, output_dir)
    example_oscillating_params(input_path, output_dir)
    example_random_glitch_sequence(input_path, output_dir)
    example_effect_animation(input_path, output_dir)
    example_colorspace_morph(input_path, output_dir)
    example_strobe_effect(input_path, output_dir)

    print("\n" + "=" * 60)
    print("All GIF animations created!")
    print(f"Check output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
