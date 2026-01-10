#!/usr/bin/env python3
"""
Basic GLIC Usage Examples

This script demonstrates the simplest ways to use GLIC.
"""

import glic

# Example 1: Simple one-liner
# glic.glitch("input.png", "output.png")

# Example 2: Using a preset
# glic.glitch("input.png", "output.png", preset="heavy_glitch")

# Example 3: Custom intensity
# glic.glitch("input.png", "output.png", intensity=0.8)

# Example 4: With effects
# glic.glitch("input.png", "output.png", effects=["scanline", "chromatic"])

# Example 5: Random glitch
# glic.random_glitch("input.png", "output.png")

# Example 6: Create multiple variations
# glic.create_variations("input.png", "output_folder/", num_variations=5)

# Example 7: Batch processing
# glic.batch_glitch("input_folder/", "output_folder/")


def main():
    """Run examples if an image path is provided."""
    import sys

    if len(sys.argv) < 2:
        print("GLIC Basic Usage Examples")
        print("=" * 40)
        print()
        print("Usage: python basic_usage.py <image_path>")
        print()
        print("Available presets:")
        for p in glic.list_presets():
            print(f"  - {p}")
        print()
        print("Available effects:")
        for e in glic.list_effects():
            print(f"  - {e}")
        return

    input_path = sys.argv[1]
    print(f"Processing: {input_path}")

    # Apply default glitch
    output = glic.glitch(input_path, intensity=0.5)
    print(f"Created: {output}")

    # Apply heavy glitch
    output = glic.glitch(
        input_path,
        input_path.replace(".png", "_heavy.png"),
        preset="heavy_glitch"
    )
    print(f"Created: {output}")

    # Apply with effects
    output = glic.glitch(
        input_path,
        input_path.replace(".png", "_effects.png"),
        intensity=0.6,
        effects=["scanline", "chromatic"]
    )
    print(f"Created: {output}")


if __name__ == "__main__":
    main()
