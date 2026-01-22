#!/usr/bin/env python3
"""
GLIC Segmentation Examples

This script demonstrates how GLIC's quad-tree segmentation works:
- Visualizing how images are divided into variable-sized blocks
- Understanding how segmentation parameters affect the glitch effect
- Using different precision values for different artistic results
- Analyzing segment distribution for optimization

Quad-tree segmentation is the core of GLIC's adaptive block sizing:
- Areas with high detail (high variance) get smaller blocks
- Areas with low detail (low variance) get larger blocks
- This creates the characteristic "blocky" glitch aesthetic
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from glic.segment import (
    Segment, segment_plane, count_segments_by_size,
    sort_segments_raster, calculate_std
)
from glic.planes import load_image, save_image, Planes
from glic.config import ColorSpace, PredictionMethod
from glic.codec import GlicCodec
from glic.config import CodecConfig, ChannelConfig


def visualize_segments(
    image: np.ndarray,
    segments: List[Segment],
    output_path: str,
    show_grid: bool = True,
    colorize_by_size: bool = True
):
    """
    Visualize segmentation by drawing segment boundaries on the image.

    Args:
        image: Original RGB image as numpy array
        segments: List of Segment objects
        output_path: Path to save visualization
        show_grid: Whether to draw grid lines
        colorize_by_size: Whether to color segments by their size
    """
    result = image.copy()

    if colorize_by_size:
        # Create size-to-color mapping
        sizes = set((s.width, s.height) for s in segments)
        size_list = sorted(sizes, key=lambda s: s[0] * s[1])

        # Color palette from small (warm) to large (cool)
        colors = [
            (255, 100, 100),  # Small - red
            (255, 180, 100),  # Orange
            (255, 255, 100),  # Yellow
            (100, 255, 100),  # Green
            (100, 255, 255),  # Cyan
            (100, 100, 255),  # Blue
            (180, 100, 255),  # Purple
            (255, 100, 255),  # Magenta - large
        ]

        size_to_color = {}
        for i, size in enumerate(size_list):
            color_idx = int(i / len(size_list) * (len(colors) - 1))
            size_to_color[size] = colors[color_idx]

        # Fill segments with semi-transparent color
        overlay = image.copy().astype(np.float32)
        for seg in segments:
            color = size_to_color.get((seg.width, seg.height), (200, 200, 200))
            y1, y2 = seg.y, min(seg.y + seg.height, image.shape[0])
            x1, x2 = seg.x, min(seg.x + seg.width, image.shape[1])
            for c in range(3):
                overlay[y1:y2, x1:x2, c] = (
                    overlay[y1:y2, x1:x2, c] * 0.6 + color[c] * 0.4
                )
        result = overlay.astype(np.uint8)

    if show_grid:
        # Draw segment boundaries
        for seg in segments:
            # Top edge
            if seg.y < result.shape[0]:
                result[seg.y, seg.x:min(seg.x + seg.width, result.shape[1])] = [255, 255, 255]
            # Left edge
            if seg.x < result.shape[1]:
                result[seg.y:min(seg.y + seg.height, result.shape[0]), seg.x] = [255, 255, 255]

    save_image(result, output_path)


def example_basic_segmentation(input_path: str, output_dir: str):
    """
    Example 1: Basic segmentation visualization.

    Shows how an image is divided into variable-sized blocks.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Segmentation Visualization")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load image and convert to grayscale for segmentation
    image = load_image(input_path)
    gray = np.mean(image, axis=2).astype(np.int32)

    # Segment with default parameters
    segments = segment_plane(
        gray,
        min_size=4,
        max_size=64,
        precision=15.0,
        prediction_method=PredictionMethod.PAETH
    )
    segments = sort_segments_raster(segments)

    print(f"\nImage size: {image.shape[1]}x{image.shape[0]}")
    print(f"Total segments: {len(segments)}")

    # Analyze segment sizes
    size_counts = count_segments_by_size(segments)
    print("\nSegment sizes:")
    for size, count in sorted(size_counts.items(), key=lambda x: x[0][0] * x[0][1]):
        print(f"  {size[0]}x{size[1]}: {count} blocks")

    # Visualize
    vis_path = str(output_path / "segmentation_basic.png")
    visualize_segments(image, segments, vis_path)
    print(f"\nVisualization saved: {vis_path}")

    return segments


def example_precision_comparison(input_path: str, output_dir: str):
    """
    Example 2: Compare different precision values.

    Lower precision = more uniform blocks (less adaptive)
    Higher precision = more adaptive to image content
    """
    print("\n" + "=" * 60)
    print("Example 2: Precision Comparison")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    gray = np.mean(image, axis=2).astype(np.int32)

    precision_values = [5.0, 15.0, 30.0, 50.0, 100.0]

    print(f"\nComparing precision values: {precision_values}")
    print("-" * 50)

    for precision in precision_values:
        segments = segment_plane(
            gray,
            min_size=4,
            max_size=64,
            precision=precision,
            prediction_method=PredictionMethod.PAETH
        )

        size_counts = count_segments_by_size(segments)
        avg_size = sum(s.width * s.height for s in segments) / len(segments)

        print(f"\nPrecision {precision:5.1f}: {len(segments):4d} segments, "
              f"avg size: {avg_size:6.1f}px, "
              f"unique sizes: {len(size_counts)}")

        # Visualize
        vis_path = str(output_path / f"precision_{int(precision):03d}.png")
        visualize_segments(image, segments, vis_path)

        # Also create glitched version
        config = CodecConfig()
        for ch in config.channels:
            ch.segmentation_precision = precision
            ch.min_block_size = 4
            ch.max_block_size = 64

        codec = GlicCodec(config)
        glitch_path = str(output_path / f"glitch_precision_{int(precision):03d}.png")
        codec.encode_decode(input_path, glitch_path)

    print(f"\nVisualizations saved to: {output_path}")


def example_block_size_comparison(input_path: str, output_dir: str):
    """
    Example 3: Compare different min/max block sizes.

    Block size affects the "resolution" of the glitch effect.
    """
    print("\n" + "=" * 60)
    print("Example 3: Block Size Comparison")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    gray = np.mean(image, axis=2).astype(np.int32)

    # Different block size configurations
    configs = [
        ("tiny", 2, 16),
        ("small", 4, 32),
        ("medium", 8, 64),
        ("large", 16, 128),
        ("huge", 32, 256),
    ]

    print(f"\nComparing block size configurations:")
    print("-" * 50)

    for name, min_size, max_size in configs:
        segments = segment_plane(
            gray,
            min_size=min_size,
            max_size=max_size,
            precision=15.0,
            prediction_method=PredictionMethod.PAETH
        )

        print(f"\n{name:8s} (min={min_size:3d}, max={max_size:3d}): "
              f"{len(segments):5d} segments")

        # Visualize segmentation
        vis_path = str(output_path / f"blocksize_{name}.png")
        visualize_segments(image, segments, vis_path, colorize_by_size=True)

        # Create glitched version
        config = CodecConfig()
        for ch in config.channels:
            ch.min_block_size = min_size
            ch.max_block_size = max_size

        codec = GlicCodec(config)
        glitch_path = str(output_path / f"glitch_blocksize_{name}.png")
        codec.encode_decode(input_path, glitch_path)

    print(f"\nVisualizations saved to: {output_path}")


def example_channel_segmentation(input_path: str, output_dir: str):
    """
    Example 4: Per-channel segmentation analysis.

    Shows how each color channel segments differently based on content.
    """
    print("\n" + "=" * 60)
    print("Example 4: Per-Channel Segmentation")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)

    # Analyze segmentation in different color spaces
    colorspaces = [
        (ColorSpace.RGB, ["Red", "Green", "Blue"]),
        (ColorSpace.YUV, ["Y (Luma)", "U (Cb)", "V (Cr)"]),
        (ColorSpace.HSB, ["Hue", "Saturation", "Brightness"]),
    ]

    for colorspace, channel_names in colorspaces:
        print(f"\n{colorspace.name} Color Space:")
        print("-" * 40)

        # Convert to color space
        planes = Planes.from_image(image, colorspace, (128, 128, 128))

        for ch in range(3):
            plane = planes.get_plane(ch)
            segments = segment_plane(
                plane,
                min_size=4,
                max_size=64,
                precision=15.0,
                prediction_method=PredictionMethod.PAETH
            )

            size_counts = count_segments_by_size(segments)
            print(f"  {channel_names[ch]:15s}: {len(segments):4d} segments, "
                  f"{len(size_counts)} unique sizes")

            # Visualize channel segmentation
            # Convert single channel to RGB for visualization
            channel_vis = np.stack([plane.astype(np.uint8)] * 3, axis=-1)
            vis_path = str(output_path / f"channel_{colorspace.name}_{ch}.png")
            visualize_segments(channel_vis, segments, vis_path)


def example_adaptive_vs_uniform(input_path: str, output_dir: str):
    """
    Example 5: Compare adaptive (quad-tree) vs uniform block segmentation.

    Demonstrates the visual difference between adaptive and fixed-size blocks.
    """
    print("\n" + "=" * 60)
    print("Example 5: Adaptive vs Uniform Segmentation")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    gray = np.mean(image, axis=2).astype(np.int32)

    # Adaptive segmentation (normal quad-tree)
    adaptive_segments = segment_plane(
        gray,
        min_size=8,
        max_size=64,
        precision=15.0,
        prediction_method=PredictionMethod.PAETH
    )

    # "Uniform" segmentation (very low precision = no splitting)
    uniform_segments = segment_plane(
        gray,
        min_size=16,
        max_size=16,  # Force uniform size
        precision=1000.0,  # Very high = never split
        prediction_method=PredictionMethod.PAETH
    )

    print(f"\nAdaptive segmentation: {len(adaptive_segments)} segments")
    print(f"Uniform segmentation:  {len(uniform_segments)} segments")

    # Visualize
    vis_adaptive = str(output_path / "adaptive_segments.png")
    vis_uniform = str(output_path / "uniform_segments.png")
    visualize_segments(image, adaptive_segments, vis_adaptive)
    visualize_segments(image, uniform_segments, vis_uniform)

    # Create glitched versions
    # Adaptive
    config_adaptive = CodecConfig()
    for ch in config_adaptive.channels:
        ch.min_block_size = 8
        ch.max_block_size = 64
        ch.segmentation_precision = 15.0

    codec = GlicCodec(config_adaptive)
    codec.encode_decode(input_path, str(output_path / "glitch_adaptive.png"))

    # Uniform
    config_uniform = CodecConfig()
    for ch in config_uniform.channels:
        ch.min_block_size = 16
        ch.max_block_size = 16
        ch.segmentation_precision = 1000.0

    codec = GlicCodec(config_uniform)
    codec.encode_decode(input_path, str(output_path / "glitch_uniform.png"))

    print(f"\nVisualizations saved to: {output_path}")


def example_segment_statistics(input_path: str, output_dir: str):
    """
    Example 6: Detailed segment statistics and analysis.

    Useful for understanding and optimizing glitch parameters.
    """
    print("\n" + "=" * 60)
    print("Example 6: Segment Statistics")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    gray = np.mean(image, axis=2).astype(np.int32)

    segments = segment_plane(
        gray,
        min_size=4,
        max_size=128,
        precision=15.0,
        prediction_method=PredictionMethod.PAETH
    )

    # Calculate statistics
    total_pixels = image.shape[0] * image.shape[1]
    segment_areas = [s.width * s.height for s in segments]

    print(f"\nImage: {image.shape[1]}x{image.shape[0]} = {total_pixels:,} pixels")
    print(f"Total segments: {len(segments)}")
    print(f"\nSegment area statistics:")
    print(f"  Min area:    {min(segment_areas):,} pixels")
    print(f"  Max area:    {max(segment_areas):,} pixels")
    print(f"  Mean area:   {np.mean(segment_areas):,.1f} pixels")
    print(f"  Median area: {np.median(segment_areas):,.1f} pixels")
    print(f"  Std dev:     {np.std(segment_areas):,.1f} pixels")

    # Coverage analysis
    covered_pixels = sum(segment_areas)
    print(f"\nCoverage: {covered_pixels:,} / {total_pixels:,} pixels "
          f"({covered_pixels/total_pixels*100:.1f}%)")

    # Size distribution
    size_counts = count_segments_by_size(segments)
    print(f"\nSize distribution ({len(size_counts)} unique sizes):")

    # Sort by area
    sorted_sizes = sorted(size_counts.items(), key=lambda x: x[0][0] * x[0][1])
    for (w, h), count in sorted_sizes:
        area = w * h
        percentage = (count * area / total_pixels) * 100
        bar = "#" * int(percentage)
        print(f"  {w:3d}x{h:<3d} ({area:5d}px): {count:4d} blocks, "
              f"{percentage:5.1f}% {bar}")

    # Save statistics to file
    stats_path = output_path / "segment_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Image: {input_path}\n")
        f.write(f"Size: {image.shape[1]}x{image.shape[0]}\n")
        f.write(f"Total segments: {len(segments)}\n")
        f.write(f"Min area: {min(segment_areas)}\n")
        f.write(f"Max area: {max(segment_areas)}\n")
        f.write(f"Mean area: {np.mean(segment_areas):.1f}\n")

    print(f"\nStatistics saved to: {stats_path}")


def print_usage():
    """Print usage information."""
    print("GLIC Segmentation Examples")
    print("=" * 40)
    print()
    print("Usage: python segmentation.py <image_path> [output_dir]")
    print()
    print("Arguments:")
    print("  image_path  Path to input image (PNG, JPG, BMP)")
    print("  output_dir  Output directory (default: ./segmentation_output)")
    print()
    print("This script demonstrates:")
    print("  1. Basic segmentation visualization")
    print("  2. Precision value comparison")
    print("  3. Block size comparison")
    print("  4. Per-channel segmentation analysis")
    print("  5. Adaptive vs uniform segmentation")
    print("  6. Segment statistics and analysis")


def main():
    """Run all segmentation examples."""
    if len(sys.argv) < 2:
        print_usage()
        return

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./segmentation_output"

    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        return

    print("GLIC Segmentation Examples")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")

    # Run all examples
    example_basic_segmentation(input_path, output_dir)
    example_precision_comparison(input_path, output_dir)
    example_block_size_comparison(input_path, output_dir)
    example_channel_segmentation(input_path, output_dir)
    example_adaptive_vs_uniform(input_path, output_dir)
    example_segment_statistics(input_path, output_dir)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print(f"Check output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
