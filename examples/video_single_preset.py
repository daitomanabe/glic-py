#!/usr/bin/env python3
"""
Video Processing with Single Preset

Process all frames of a video with a specific GLIC preset.

Requirements:
    pip install opencv-python

Usage:
    python video_single_preset.py input.mp4 output.mp4 --preset vhs
    python video_single_preset.py input.mp4 output.mp4 --preset acid --intensity 0.8
    python video_single_preset.py input.mp4 output/ --frames --preset heavy_glitch
"""

import argparse
import sys
from pathlib import Path

# Add parent directory for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import cv2
except ImportError:
    print("Error: opencv-python is required")
    print("Install with: pip install opencv-python")
    sys.exit(1)

import glic
from glic import glitch_image


def process_video_to_video(
    input_path: str,
    output_path: str,
    preset: str = "default",
    intensity: float = 0.5,
    effects: list = None,
    show_progress: bool = True
):
    """
    Process video and output as video file.

    Args:
        input_path: Input video file path
        output_path: Output video file path
        preset: GLIC preset name
        intensity: Glitch intensity (0.0-1.0)
        effects: List of effect names to apply
        show_progress: Show progress during processing
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {input_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Preset: {preset}")
    print(f"Intensity: {intensity}")
    print()

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply GLIC glitch
        glitched = glitch_image(
            frame_rgb,
            preset=preset,
            intensity=intensity,
            effects=effects
        )

        # Convert RGB back to BGR
        glitched_bgr = cv2.cvtColor(glitched, cv2.COLOR_RGB2BGR)

        # Write frame
        out.write(glitched_bgr)

        frame_count += 1
        if show_progress:
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing: {frame_count}/{total_frames} ({progress:.1f}%)", end="")

    cap.release()
    out.release()

    if show_progress:
        print()

    print(f"Output: {output_path}")
    print(f"Processed {frame_count} frames")


def process_video_to_frames(
    input_path: str,
    output_dir: str,
    preset: str = "default",
    intensity: float = 0.5,
    effects: list = None,
    format: str = "png",
    show_progress: bool = True
):
    """
    Process video and output as individual frame images.

    Args:
        input_path: Input video file path
        output_dir: Output directory for frames
        preset: GLIC preset name
        intensity: Glitch intensity (0.0-1.0)
        effects: List of effect names to apply
        format: Output image format (png, jpg)
        show_progress: Show progress during processing
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {input_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Preset: {preset}")
    print(f"Output directory: {output_dir}")
    print()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply GLIC glitch
        glitched = glitch_image(
            frame_rgb,
            preset=preset,
            intensity=intensity,
            effects=effects
        )

        # Convert RGB back to BGR for saving
        glitched_bgr = cv2.cvtColor(glitched, cv2.COLOR_RGB2BGR)

        # Save frame
        frame_path = output_path / f"frame_{frame_count:06d}.{format}"
        cv2.imwrite(str(frame_path), glitched_bgr)

        frame_count += 1
        if show_progress:
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing: {frame_count}/{total_frames} ({progress:.1f}%)", end="")

    cap.release()

    if show_progress:
        print()

    print(f"Saved {frame_count} frames to {output_dir}")

    # Save metadata
    metadata_path = output_path / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"source: {input_path}\n")
        f.write(f"preset: {preset}\n")
        f.write(f"intensity: {intensity}\n")
        f.write(f"fps: {fps}\n")
        f.write(f"width: {width}\n")
        f.write(f"height: {height}\n")
        f.write(f"total_frames: {frame_count}\n")

    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process video with GLIC glitch effect using a single preset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process video to video
    python video_single_preset.py input.mp4 output.mp4 --preset vhs

    # Process with custom intensity
    python video_single_preset.py input.mp4 output.mp4 --preset acid --intensity 0.8

    # Output as individual frames
    python video_single_preset.py input.mp4 frames/ --frames --preset heavy_glitch

    # Add effects
    python video_single_preset.py input.mp4 output.mp4 --preset minimal --effect scanline --effect chromatic

Available presets:
    default, heavy_glitch, minimal, high_quality, vhs, datamosh, acid,
    noise, dreamy, blocks, corrupted, mosaic, edges, interference,
    analog, displacement, color_waves, pixelated, spiral, chromatic
        """
    )

    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file or directory (with --frames)")
    parser.add_argument("--preset", "-p", default="default", help="GLIC preset name")
    parser.add_argument("--intensity", "-i", type=float, default=0.5, help="Glitch intensity (0.0-1.0)")
    parser.add_argument("--effect", "-e", action="append", help="Add post-processing effect")
    parser.add_argument("--frames", "-f", action="store_true", help="Output as individual frames")
    parser.add_argument("--format", default="png", choices=["png", "jpg"], help="Frame output format")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        if args.frames:
            process_video_to_frames(
                args.input,
                args.output,
                preset=args.preset,
                intensity=args.intensity,
                effects=args.effect,
                format=args.format,
                show_progress=not args.quiet
            )
        else:
            process_video_to_video(
                args.input,
                args.output,
                preset=args.preset,
                intensity=args.intensity,
                effects=args.effect,
                show_progress=not args.quiet
            )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
