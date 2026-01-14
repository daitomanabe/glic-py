#!/usr/bin/env python3
"""
Video Processing with Multiple Presets

Process a video with all presets from a specified folder.
Each preset generates a separate output video/frames.

Requirements:
    pip install opencv-python

Usage:
    # Use built-in presets
    python video_multi_preset.py input.mp4 output_dir/

    # Use custom preset folder
    python video_multi_preset.py input.mp4 output_dir/ --presets-dir ./my_presets/

    # Output as frames instead of videos
    python video_multi_preset.py input.mp4 output_dir/ --frames

    # Process with specific presets only
    python video_multi_preset.py input.mp4 output_dir/ --only vhs acid datamosh
"""

import argparse
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
from glic.preset_loader import PresetLoader, get_default_presets_dir


def get_presets_from_dir(presets_dir: str) -> dict:
    """
    Get all presets from a directory.

    Args:
        presets_dir: Path to presets directory

    Returns:
        Dictionary of preset_name -> preset_config
    """
    presets = {}
    presets_path = Path(presets_dir)

    if not presets_path.exists():
        print(f"Warning: Presets directory not found: {presets_dir}")
        return presets

    # Load .json presets
    for json_file in presets_path.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                config = json.load(f)
            preset_name = json_file.stem
            presets[preset_name] = config
            print(f"  Loaded: {preset_name}")
        except Exception as e:
            print(f"  Warning: Failed to load {json_file}: {e}")

    # Load .glic presets
    for glic_file in presets_path.glob("*.glic"):
        try:
            with open(glic_file, "r") as f:
                config = json.load(f)
            preset_name = glic_file.stem
            presets[preset_name] = config
            print(f"  Loaded: {preset_name}")
        except Exception as e:
            print(f"  Warning: Failed to load {glic_file}: {e}")

    return presets


def get_builtin_presets() -> list:
    """Get list of built-in preset names."""
    return glic.list_presets()


def process_video_with_preset(
    input_path: str,
    output_path: str,
    preset_name: str,
    intensity: float = 0.5,
    as_frames: bool = False,
    frame_format: str = "png"
) -> dict:
    """
    Process a video with a single preset.

    Args:
        input_path: Input video path
        output_path: Output video or directory path
        preset_name: Name of the preset
        intensity: Glitch intensity
        as_frames: Output as frames instead of video
        frame_format: Frame output format

    Returns:
        Result dictionary with status and info
    """
    result = {
        "preset": preset_name,
        "output": output_path,
        "status": "success",
        "frames": 0,
        "error": None
    }

    try:
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if as_frames:
            # Output as frames
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                glitched = glitch_image(frame_rgb, preset=preset_name, intensity=intensity)
                glitched_bgr = cv2.cvtColor(glitched, cv2.COLOR_RGB2BGR)

                frame_path = output_dir / f"frame_{frame_count:06d}.{frame_format}"
                cv2.imwrite(str(frame_path), glitched_bgr)
                frame_count += 1

            result["frames"] = frame_count

        else:
            # Output as video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                glitched = glitch_image(frame_rgb, preset=preset_name, intensity=intensity)
                glitched_bgr = cv2.cvtColor(glitched, cv2.COLOR_RGB2BGR)

                out.write(glitched_bgr)
                frame_count += 1

            out.release()
            result["frames"] = frame_count

        cap.release()

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def process_video_all_presets(
    input_path: str,
    output_dir: str,
    presets_dir: str = None,
    only_presets: list = None,
    intensity: float = 0.5,
    as_frames: bool = False,
    frame_format: str = "png",
    parallel: bool = False,
    max_workers: int = None
):
    """
    Process video with all presets.

    Args:
        input_path: Input video path
        output_dir: Output directory
        presets_dir: Custom presets directory (None for built-in)
        only_presets: Process only these presets
        intensity: Glitch intensity
        as_frames: Output as frames
        frame_format: Frame format
        parallel: Process in parallel
        max_workers: Number of parallel workers
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get presets
    if presets_dir:
        print(f"Loading presets from: {presets_dir}")
        custom_presets = get_presets_from_dir(presets_dir)
        preset_names = list(custom_presets.keys())
    else:
        print("Using built-in presets")
        preset_names = get_builtin_presets()

    # Filter presets if specified
    if only_presets:
        preset_names = [p for p in preset_names if p in only_presets]

    if not preset_names:
        print("Error: No presets found")
        return

    print(f"\nInput: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Presets to process: {len(preset_names)}")
    print(f"Intensity: {intensity}")
    print(f"Output mode: {'frames' if as_frames else 'video'}")
    print()

    results = []

    if parallel:
        # Parallel processing
        workers = max_workers or min(len(preset_names), multiprocessing.cpu_count())
        print(f"Processing in parallel with {workers} workers...")
        print()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}

            for preset_name in preset_names:
                if as_frames:
                    preset_output = str(output_path / preset_name)
                else:
                    preset_output = str(output_path / f"{preset_name}.mp4")

                future = executor.submit(
                    process_video_with_preset,
                    input_path,
                    preset_output,
                    preset_name,
                    intensity,
                    as_frames,
                    frame_format
                )
                futures[future] = preset_name

            for future in as_completed(futures):
                preset_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "OK" if result["status"] == "success" else "FAILED"
                    print(f"  [{status}] {preset_name}: {result['frames']} frames")
                except Exception as e:
                    print(f"  [FAILED] {preset_name}: {e}")
                    results.append({
                        "preset": preset_name,
                        "status": "failed",
                        "error": str(e)
                    })

    else:
        # Sequential processing
        for i, preset_name in enumerate(preset_names):
            print(f"[{i+1}/{len(preset_names)}] Processing with preset: {preset_name}")

            if as_frames:
                preset_output = str(output_path / preset_name)
            else:
                preset_output = str(output_path / f"{preset_name}.mp4")

            result = process_video_with_preset(
                input_path,
                preset_output,
                preset_name,
                intensity,
                as_frames,
                frame_format
            )
            results.append(result)

            if result["status"] == "success":
                print(f"    -> {result['output']} ({result['frames']} frames)")
            else:
                print(f"    -> FAILED: {result['error']}")

    # Summary
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"Total: {len(results)}")
    print(f"Success: {len(success)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed presets:")
        for r in failed:
            print(f"  - {r['preset']}: {r.get('error', 'Unknown error')}")

    # Save results summary
    summary_path = output_path / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "input": input_path,
            "output_dir": output_dir,
            "intensity": intensity,
            "as_frames": as_frames,
            "total_presets": len(preset_names),
            "success": len(success),
            "failed": len(failed),
            "results": results
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process video with multiple GLIC presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use all built-in presets
    python video_multi_preset.py input.mp4 output_dir/

    # Use custom preset folder
    python video_multi_preset.py input.mp4 output_dir/ --presets-dir ./my_presets/

    # Output as frames
    python video_multi_preset.py input.mp4 output_dir/ --frames

    # Process specific presets only
    python video_multi_preset.py input.mp4 output_dir/ --only vhs acid datamosh

    # Parallel processing
    python video_multi_preset.py input.mp4 output_dir/ --parallel --workers 4

    # Custom intensity
    python video_multi_preset.py input.mp4 output_dir/ --intensity 0.8

Creating custom presets:
    Custom presets should be JSON files with GLIC configuration.
    Place them in a folder and use --presets-dir to specify it.

    Example preset (my_preset.json):
    {
        "color_space": "YUV",
        "channels": [
            {"prediction_method": "SPIRAL", "quantization_value": 150}
        ]
    }
        """
    )

    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--presets-dir", "-d", help="Custom presets directory")
    parser.add_argument("--only", nargs="+", help="Process only these presets")
    parser.add_argument("--intensity", "-i", type=float, default=0.5, help="Glitch intensity")
    parser.add_argument("--frames", "-f", action="store_true", help="Output as frames")
    parser.add_argument("--format", default="png", choices=["png", "jpg"], help="Frame format")
    parser.add_argument("--parallel", "-p", action="store_true", help="Process in parallel")
    parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        process_video_all_presets(
            args.input,
            args.output,
            presets_dir=args.presets_dir,
            only_presets=args.only,
            intensity=args.intensity,
            as_frames=args.frames,
            frame_format=args.format,
            parallel=args.parallel,
            max_workers=args.workers
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
