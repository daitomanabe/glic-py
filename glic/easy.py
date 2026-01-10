"""
Easy-to-use API for GLIC.

This module provides simple one-liner functions for common glitch operations.

Examples:
    >>> import glic
    >>> glic.glitch("input.png", "output.png")
    >>> glic.glitch("input.png", "output.png", preset="heavy_glitch")
    >>> glic.glitch("input.png", "output.png", intensity=0.8)
    >>> glic.batch_glitch("input_folder", "output_folder")
"""

import os
from pathlib import Path
from typing import Optional, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .codec import GlicCodec
from .config import (
    CodecConfig, ColorSpace, PredictionMethod, TransformType,
    WaveletType, EffectType, EffectConfig
)
from .preset_loader import PresetLoader
from .planes import load_image, save_image
from .effects import create_effect_config, apply_effects


def glitch(
    input_path: str,
    output_path: Optional[str] = None,
    preset: Optional[str] = None,
    intensity: float = 0.5,
    colorspace: Optional[str] = None,
    prediction: Optional[str] = None,
    effects: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Apply glitch effect to an image with simple parameters.

    Args:
        input_path: Path to input image
        output_path: Path for output (default: input_glitched.png)
        preset: Preset name ('heavy_glitch', 'minimal', 'color_waves', etc.)
        intensity: Glitch intensity 0.0-1.0 (default: 0.5)
        colorspace: Color space name (e.g., 'YUV', 'HSB', 'HWB')
        prediction: Prediction method (e.g., 'SPIRAL', 'WAVE', 'PAETH')
        effects: List of effect names ('scanline', 'chromatic', 'pixelate', etc.)
        seed: Random seed for reproducible results

    Returns:
        Path to output file

    Examples:
        >>> glitch("photo.png")
        'photo_glitched.png'

        >>> glitch("photo.png", preset="heavy_glitch")
        'photo_glitched.png'

        >>> glitch("photo.png", intensity=0.8, effects=["scanline", "chromatic"])
        'photo_glitched.png'
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    # Generate output path if not provided
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_glitched{p.suffix}")

    # Load preset or create config
    loader = PresetLoader()
    if preset:
        config = loader.load_preset(preset)
        if config is None:
            config = CodecConfig()
    else:
        config = CodecConfig()

    # Apply intensity scaling
    quant = int(50 + intensity * 150)  # 50-200 range
    scale = int(intensity * 50)  # 0-50 range

    for ch in config.channels:
        ch.quantization_value = quant
        ch.transform_scale = scale

    # Apply colorspace
    if colorspace:
        try:
            config.color_space = ColorSpace[colorspace.upper()]
        except KeyError:
            pass

    # Apply prediction method
    if prediction:
        try:
            pred = PredictionMethod[prediction.upper()]
            for ch in config.channels:
                ch.prediction_method = pred
        except KeyError:
            pass

    # Apply effects
    if effects:
        for effect_name in effects:
            try:
                effect_type = EffectType[effect_name.upper()]
                config.effects.append(create_effect_config(
                    effect_type,
                    intensity=int(intensity * 100)
                ))
            except KeyError:
                pass

    # Process
    codec = GlicCodec(config)
    codec.encode_decode(input_path, output_path)

    return output_path


def glitch_image(
    image: np.ndarray,
    preset: Optional[str] = None,
    intensity: float = 0.5,
    colorspace: Optional[str] = None,
    prediction: Optional[str] = None,
    effects: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Apply glitch effect to a numpy array image.

    Args:
        image: Input image as numpy array (H, W, 3) RGB
        preset: Preset name
        intensity: Glitch intensity 0.0-1.0
        colorspace: Color space name
        prediction: Prediction method
        effects: List of effect names

    Returns:
        Glitched image as numpy array

    Examples:
        >>> import numpy as np
        >>> from PIL import Image
        >>> img = np.array(Image.open("photo.png"))
        >>> result = glitch_image(img, intensity=0.7)
        >>> Image.fromarray(result).save("output.png")
    """
    import tempfile
    import os

    # Save to temp file, process, load back
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")

        save_image(image, input_path)
        glitch(
            input_path, output_path,
            preset=preset, intensity=intensity,
            colorspace=colorspace, prediction=prediction,
            effects=effects
        )
        return load_image(output_path)


def batch_glitch(
    input_dir: str,
    output_dir: Optional[str] = None,
    preset: Optional[str] = None,
    intensity: float = 0.5,
    effects: Optional[List[str]] = None,
    pattern: str = "*.png",
    max_workers: int = 4,
    progress: bool = True,
) -> List[str]:
    """
    Apply glitch effect to all images in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory (default: input_dir/glitched)
        preset: Preset name
        intensity: Glitch intensity 0.0-1.0
        effects: List of effect names
        pattern: Glob pattern for input files (default: "*.png")
        max_workers: Number of parallel workers (default: 4)
        progress: Show progress (default: True)

    Returns:
        List of output file paths

    Examples:
        >>> batch_glitch("photos/", "output/", preset="heavy_glitch")
        ['output/photo1.png', 'output/photo2.png', ...]

        >>> batch_glitch("photos/", pattern="*.jpg", intensity=0.8)
        [...]
    """
    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path / "glitched"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    input_files = list(input_path.glob(pattern))

    # Also check for common image extensions
    if not input_files and pattern == "*.png":
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG", "*.JPEG"]:
            input_files.extend(input_path.glob(ext))

    if not input_files:
        print(f"No files found matching '{pattern}' in {input_dir}")
        return []

    output_files = []
    total = len(input_files)

    def process_file(input_file: Path) -> str:
        out_file = output_path / f"{input_file.stem}_glitched.png"
        glitch(
            str(input_file), str(out_file),
            preset=preset, intensity=intensity, effects=effects
        )
        return str(out_file)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in input_files}

        for i, future in enumerate(as_completed(futures)):
            input_file = futures[future]
            try:
                output_file = future.result()
                output_files.append(output_file)
                if progress:
                    print(f"[{i+1}/{total}] {input_file.name} -> {Path(output_file).name}")
            except Exception as e:
                if progress:
                    print(f"[{i+1}/{total}] Error processing {input_file.name}: {e}")

    return output_files


def list_presets() -> List[str]:
    """Get list of available preset names."""
    loader = PresetLoader()
    return loader.list_presets()


def list_colorspaces() -> List[str]:
    """Get list of available color space names."""
    return [cs.name for cs in ColorSpace]


def list_predictions() -> List[str]:
    """Get list of available prediction method names."""
    return [pm.name for pm in PredictionMethod]


def list_effects() -> List[str]:
    """Get list of available effect names."""
    return [et.name.lower() for et in EffectType]


def random_glitch(
    input_path: str,
    output_path: Optional[str] = None,
    intensity: float = 0.5,
) -> str:
    """
    Apply random glitch effect with randomized parameters.

    Args:
        input_path: Path to input image
        output_path: Path for output
        intensity: Base intensity level 0.0-1.0

    Returns:
        Path to output file
    """
    import random

    # Random colorspace
    colorspace = random.choice(["RGB", "YUV", "HSB", "HWB", "OHTA", "LAB"])

    # Random prediction
    predictions = ["PAETH", "SPIRAL", "WAVE", "GRADIENT", "RADIAL", "NOISE", "CHECKERBOARD"]
    prediction = random.choice(predictions)

    # Random effects (0-2 effects)
    all_effects = ["scanline", "chromatic", "pixelate", "dither", "posterize", "glitch_shift"]
    num_effects = random.randint(0, 2)
    effects = random.sample(all_effects, num_effects) if num_effects > 0 else None

    # Random intensity variation
    actual_intensity = intensity * random.uniform(0.7, 1.3)
    actual_intensity = max(0.1, min(1.0, actual_intensity))

    return glitch(
        input_path, output_path,
        intensity=actual_intensity,
        colorspace=colorspace,
        prediction=prediction,
        effects=effects
    )


def create_variations(
    input_path: str,
    output_dir: str,
    num_variations: int = 5,
    intensity_range: Tuple[float, float] = (0.3, 0.8),
) -> List[str]:
    """
    Create multiple glitch variations of an image.

    Args:
        input_path: Path to input image
        output_dir: Output directory
        num_variations: Number of variations to create
        intensity_range: Min and max intensity (default: 0.3-0.8)

    Returns:
        List of output file paths
    """
    import random

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_name = Path(input_path).stem
    output_files = []

    presets = list_presets()
    predictions = ["PAETH", "SPIRAL", "WAVE", "GRADIENT", "RADIAL", "NOISE"]

    for i in range(num_variations):
        intensity = random.uniform(*intensity_range)

        # Vary parameters
        preset = random.choice(presets) if random.random() > 0.5 else None
        prediction = random.choice(predictions) if preset is None else None

        out_file = output_path / f"{input_name}_v{i+1:02d}.png"

        glitch(
            input_path, str(out_file),
            preset=preset,
            intensity=intensity,
            prediction=prediction
        )
        output_files.append(str(out_file))
        print(f"Created variation {i+1}/{num_variations}: {out_file.name}")

    return output_files


# Convenience function for quick testing
def demo(image_path: Optional[str] = None):
    """
    Run a quick demo of GLIC capabilities.

    Args:
        image_path: Optional path to test image
    """
    print("GLIC - GLitch Image Codec Demo")
    print("=" * 40)

    print("\nAvailable presets:")
    for p in list_presets():
        print(f"  - {p}")

    print("\nAvailable effects:")
    for e in list_effects():
        print(f"  - {e}")

    print("\nColor spaces:", ", ".join(list_colorspaces()[:8]), "...")
    print("Predictions:", ", ".join(list_predictions()[:8]), "...")

    if image_path:
        print(f"\nProcessing {image_path}...")
        output = glitch(image_path, intensity=0.6, effects=["scanline"])
        print(f"Output: {output}")
    else:
        print("\nTo process an image:")
        print("  >>> import glic")
        print('  >>> glic.glitch("input.png", intensity=0.6)')
