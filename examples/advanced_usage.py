#!/usr/bin/env python3
"""
Advanced GLIC Usage Examples

This script demonstrates advanced configuration and usage patterns.
"""

import sys
from pathlib import Path

# Add parent directory for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from glic import (
    GlicCodec, CodecConfig, ColorSpace, PredictionMethod,
    TransformType, WaveletType, EffectType
)
from glic.effects import create_effect_config
from glic.preset_loader import PresetLoader


def example_custom_config():
    """Create a fully customized configuration."""
    config = CodecConfig()

    # Set color space
    config.color_space = ColorSpace.YUV

    # Configure each channel differently
    # Channel 0 (Y - luminance)
    config.channels[0].prediction_method = PredictionMethod.PAETH
    config.channels[0].quantization_value = 80  # Less quantization for luminance
    config.channels[0].transform_type = TransformType.FWT
    config.channels[0].wavelet_type = WaveletType.SYM8
    config.channels[0].transform_scale = 15

    # Channel 1 (U - chrominance)
    config.channels[1].prediction_method = PredictionMethod.SPIRAL
    config.channels[1].quantization_value = 150
    config.channels[1].transform_type = TransformType.WPT
    config.channels[1].wavelet_type = WaveletType.HAAR
    config.channels[1].transform_scale = 40

    # Channel 2 (V - chrominance)
    config.channels[2].prediction_method = PredictionMethod.WAVE
    config.channels[2].quantization_value = 150
    config.channels[2].transform_type = TransformType.WPT
    config.channels[2].wavelet_type = WaveletType.DB4
    config.channels[2].transform_scale = 40

    # Add post-processing effects
    config.effects.append(create_effect_config(
        EffectType.SCANLINE,
        intensity=60,
        block_size=2
    ))

    config.effects.append(create_effect_config(
        EffectType.CHROMATIC,
        intensity=40,
        block_size=3
    ))

    return config


def example_save_load_preset():
    """Save and load custom presets."""
    loader = PresetLoader()

    # Create custom config
    config = example_custom_config()

    # Save as JSON preset
    preset_path = loader.save_preset("my_custom_preset", config, ".")
    print(f"Saved preset to: {preset_path}")

    # Load it back
    loaded_config = loader.load_json_preset(preset_path)
    print(f"Loaded preset color space: {loaded_config.color_space}")


def example_numpy_integration():
    """Work with numpy arrays directly."""
    import numpy as np
    from glic import glitch_image

    # Create a test image (gradient)
    height, width = 256, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Red gradient
    image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    # Green gradient
    image[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    # Blue constant
    image[:, :, 2] = 128

    # Apply glitch
    result = glitch_image(image, intensity=0.7, prediction="SPIRAL")

    print(f"Input shape: {image.shape}, Output shape: {result.shape}")
    return result


def example_animation_frames():
    """Create a series of frames with varying glitch intensity."""
    import glic

    def create_frames(input_path, output_dir, num_frames=30):
        """Create animated frames with progressive glitch intensity."""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for i in range(num_frames):
            # Oscillating intensity
            import math
            t = i / num_frames
            intensity = 0.3 + 0.4 * math.sin(t * math.pi * 2)

            output_path = f"{output_dir}/frame_{i:03d}.png"
            glic.glitch(
                input_path, output_path,
                intensity=intensity,
                prediction="SPIRAL"
            )
            print(f"Frame {i+1}/{num_frames}")

    print("Example: Animation frames")
    print("Call create_frames('input.png', 'frames/', 30)")


def example_compare_colorspaces():
    """Compare different color spaces."""
    import glic

    colorspaces = ["RGB", "YUV", "HSB", "HWB", "LAB", "OHTA"]

    def compare(input_path, output_dir):
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for cs in colorspaces:
            output_path = f"{output_dir}/colorspace_{cs}.png"
            glic.glitch(
                input_path, output_path,
                colorspace=cs,
                intensity=0.6
            )
            print(f"Created: {output_path}")

    print("Example: Compare color spaces")
    print(f"Available color spaces: {colorspaces}")


def example_compare_predictions():
    """Compare different prediction methods."""
    import glic

    predictions = [
        "PAETH", "SPIRAL", "WAVE", "GRADIENT",
        "RADIAL", "NOISE", "CHECKERBOARD", "EDGE"
    ]

    def compare(input_path, output_dir):
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for pred in predictions:
            output_path = f"{output_dir}/prediction_{pred}.png"
            glic.glitch(
                input_path, output_path,
                prediction=pred,
                intensity=0.6
            )
            print(f"Created: {output_path}")

    print("Example: Compare predictions")
    print(f"Available predictions: {predictions}")


def main():
    print("GLIC Advanced Usage Examples")
    print("=" * 40)
    print()

    print("1. Custom Configuration")
    config = example_custom_config()
    print(f"   Color space: {config.color_space}")
    print(f"   Effects: {len(config.effects)}")
    print()

    print("2. Save/Load Presets")
    example_save_load_preset()
    print()

    print("3. NumPy Integration")
    result = example_numpy_integration()
    print()

    print("4. Animation Frames")
    example_animation_frames()
    print()

    print("5. Compare Color Spaces")
    example_compare_colorspaces()
    print()

    print("6. Compare Predictions")
    example_compare_predictions()


if __name__ == "__main__":
    main()
