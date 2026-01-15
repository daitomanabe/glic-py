#!/usr/bin/env python3
"""
Random Preset Exploration

Generate multiple glitched images with random presets and parameters
for easy comparison and selection of the best results.

Usage:
    # Generate 20 random variations
    python random_exploration.py input.png output_dir/

    # Generate 50 variations
    python random_exploration.py input.png output_dir/ --count 50

    # Use only specific presets
    python random_exploration.py input.png output_dir/ --presets vhs acid datamosh

    # Full random (random parameters, not just presets)
    python random_exploration.py input.png output_dir/ --full-random

    # Generate HTML gallery
    python random_exploration.py input.png output_dir/ --gallery
"""

import argparse
import sys
import json
import random
from pathlib import Path
from datetime import datetime

# Add parent directory for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import glic
from glic import glitch_image, GlicCodec, CodecConfig
from glic.config import ColorSpace, PredictionMethod, WaveletType, TransformType, EffectType
from glic.effects import create_effect_config

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: pillow and numpy are required")
    print("Install with: pip install pillow numpy")
    sys.exit(1)


# Available options for random generation
PRESETS = [
    "default", "heavy_glitch", "minimal", "high_quality", "vhs",
    "datamosh", "acid", "noise", "dreamy", "blocks", "corrupted",
    "mosaic", "edges", "interference", "analog", "displacement",
    "color_waves", "pixelated", "spiral", "chromatic"
]

COLORSPACES = [cs.name for cs in ColorSpace]
PREDICTIONS = [pm.name for pm in PredictionMethod]
WAVELETS = [wt.name for wt in WaveletType]
EFFECTS = [et.name.lower() for et in EffectType]


def generate_random_preset_variation(
    image: np.ndarray,
    presets: list = None,
    seed: int = None
) -> tuple:
    """
    Generate a variation using a random preset.

    Returns:
        Tuple of (result_image, params_dict)
    """
    if seed is not None:
        random.seed(seed)

    available_presets = presets or PRESETS
    preset = random.choice(available_presets)
    intensity = random.uniform(0.3, 0.9)

    # Randomly add effects
    effects = None
    if random.random() > 0.5:
        num_effects = random.randint(1, 2)
        effects = random.sample(EFFECTS, num_effects)

    result = glitch_image(
        image,
        preset=preset,
        intensity=intensity,
        effects=effects
    )

    params = {
        "type": "preset",
        "preset": preset,
        "intensity": round(intensity, 3),
        "effects": effects,
        "seed": seed
    }

    return result, params


def generate_full_random_variation(
    image: np.ndarray,
    seed: int = None
) -> tuple:
    """
    Generate a variation with fully random parameters.

    Returns:
        Tuple of (result_image, params_dict)
    """
    if seed is not None:
        random.seed(seed)

    # Random parameters
    colorspace = random.choice(COLORSPACES)
    prediction = random.choice(PREDICTIONS)
    quantization = random.randint(50, 200)
    intensity = random.uniform(0.3, 0.9)

    # Random wavelet settings
    use_wavelet = random.random() > 0.3
    wavelet = random.choice(WAVELETS) if use_wavelet else None
    transform_scale = random.randint(10, 50) if use_wavelet else None

    # Random effects
    effects = None
    if random.random() > 0.4:
        num_effects = random.randint(1, 3)
        effects = random.sample(EFFECTS, min(num_effects, len(EFFECTS)))

    result = glitch_image(
        image,
        colorspace=colorspace,
        prediction=prediction,
        quantization=quantization,
        intensity=intensity,
        wavelet=wavelet,
        transform_scale=transform_scale,
        effects=effects
    )

    params = {
        "type": "full_random",
        "colorspace": colorspace,
        "prediction": prediction,
        "quantization": quantization,
        "intensity": round(intensity, 3),
        "wavelet": wavelet,
        "transform_scale": transform_scale,
        "effects": effects,
        "seed": seed
    }

    return result, params


def generate_variations(
    input_path: str,
    output_dir: str,
    count: int = 20,
    presets: list = None,
    full_random: bool = False,
    base_seed: int = None,
    show_progress: bool = True
) -> list:
    """
    Generate multiple random variations.

    Args:
        input_path: Input image path
        output_dir: Output directory
        count: Number of variations to generate
        presets: List of presets to use (None for all)
        full_random: Use fully random parameters
        base_seed: Base seed for reproducibility
        show_progress: Show progress

    Returns:
        List of result dictionaries
    """
    # Load input image
    img = Image.open(input_path)
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy original for reference
    original_path = output_path / "000_original.png"
    Image.fromarray(img_array).save(original_path)

    results = []

    if base_seed is None:
        base_seed = random.randint(0, 999999)

    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Generating {count} variations...")
    print(f"Mode: {'Full Random' if full_random else 'Preset-based'}")
    print(f"Base seed: {base_seed}")
    print()

    for i in range(count):
        seed = base_seed + i

        try:
            if full_random:
                result_img, params = generate_full_random_variation(img_array, seed)
            else:
                result_img, params = generate_random_preset_variation(img_array, presets, seed)

            # Save image
            filename = f"{i+1:03d}_{params.get('preset', 'random')}.png"
            filepath = output_path / filename
            Image.fromarray(result_img).save(filepath)

            result = {
                "index": i + 1,
                "filename": filename,
                "filepath": str(filepath),
                "params": params
            }
            results.append(result)

            if show_progress:
                desc = params.get('preset', f"{params.get('colorspace')}/{params.get('prediction')}")
                print(f"  [{i+1:3d}/{count}] {filename} - {desc}")

        except Exception as e:
            print(f"  [{i+1:3d}/{count}] ERROR: {e}")
            results.append({
                "index": i + 1,
                "filename": None,
                "error": str(e),
                "params": params if 'params' in dir() else {}
            })

    return results


def generate_html_gallery(
    output_dir: str,
    results: list,
    input_path: str,
    title: str = "GLIC Random Exploration"
):
    """
    Generate an HTML gallery for easy comparison.
    """
    output_path = Path(output_dir)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #fff;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
        }}
        .info {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .controls {{
            text-align: center;
            margin-bottom: 20px;
            position: sticky;
            top: 0;
            background: #1a1a1a;
            padding: 10px;
            z-index: 100;
        }}
        .controls button {{
            background: #333;
            color: #fff;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            cursor: pointer;
            border-radius: 5px;
        }}
        .controls button:hover {{
            background: #444;
        }}
        .controls button.active {{
            background: #0066cc;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .gallery.large {{
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
        }}
        .card {{
            background: #2a2a2a;
            border-radius: 10px;
            overflow: hidden;
            transition: transform 0.2s;
        }}
        .card:hover {{
            transform: scale(1.02);
        }}
        .card.favorite {{
            box-shadow: 0 0 0 3px #ffcc00;
        }}
        .card img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }}
        .card-info {{
            padding: 15px;
        }}
        .card-title {{
            font-weight: bold;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-params {{
            font-size: 12px;
            color: #888;
            font-family: monospace;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        .star-btn {{
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            opacity: 0.3;
        }}
        .star-btn.active {{
            opacity: 1;
        }}
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        .modal.show {{
            display: flex;
        }}
        .modal img {{
            max-width: 95%;
            max-height: 95%;
        }}
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 30px;
            font-size: 40px;
            color: #fff;
            cursor: pointer;
        }}
        .favorites-list {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }}
        .favorites-list.show {{
            display: block;
        }}
        .favorites-list h3 {{
            margin-top: 0;
        }}
        .favorites-list pre {{
            background: #1a1a1a;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="info">
        Source: {Path(input_path).name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Total: {len(results)} variations
    </div>

    <div class="controls">
        <button onclick="toggleSize()" id="sizeBtn">Large View</button>
        <button onclick="showFavorites()" id="favBtn">Show Favorites (0)</button>
        <button onclick="copyFavorites()">Copy Favorites JSON</button>
    </div>

    <div class="favorites-list" id="favoritesList">
        <h3>Favorite Parameters</h3>
        <pre id="favoritesJson">[]</pre>
    </div>

    <div class="gallery" id="gallery">
        <div class="card">
            <img src="000_original.png" onclick="showModal(this.src)" alt="Original">
            <div class="card-info">
                <div class="card-title">
                    <span>Original</span>
                </div>
                <div class="card-params">Source image</div>
            </div>
        </div>
"""

    for result in results:
        if result.get("filename"):
            params = result["params"]
            params_str = json.dumps(params, indent=2)

            # Create display string
            if params.get("type") == "preset":
                display = f"Preset: {params.get('preset')}\\nIntensity: {params.get('intensity')}"
                if params.get("effects"):
                    display += f"\\nEffects: {', '.join(params['effects'])}"
            else:
                display = f"Colorspace: {params.get('colorspace')}\\nPrediction: {params.get('prediction')}\\nQuantization: {params.get('quantization')}"
                if params.get("wavelet"):
                    display += f"\\nWavelet: {params.get('wavelet')}"

            html += f"""
        <div class="card" data-index="{result['index']}" data-params='{json.dumps(params)}'>
            <img src="{result['filename']}" onclick="showModal(this.src)" alt="{result['filename']}">
            <div class="card-info">
                <div class="card-title">
                    <span>#{result['index']:03d}</span>
                    <button class="star-btn" onclick="toggleFavorite(this)">⭐</button>
                </div>
                <div class="card-params">{display}</div>
            </div>
        </div>
"""

    html += """
    </div>

    <div class="modal" id="modal" onclick="hideModal()">
        <span class="modal-close">&times;</span>
        <img id="modalImg" src="" alt="Preview">
    </div>

    <script>
        let favorites = [];

        function toggleSize() {
            const gallery = document.getElementById('gallery');
            const btn = document.getElementById('sizeBtn');
            gallery.classList.toggle('large');
            btn.textContent = gallery.classList.contains('large') ? 'Normal View' : 'Large View';
        }

        function showModal(src) {
            document.getElementById('modalImg').src = src;
            document.getElementById('modal').classList.add('show');
        }

        function hideModal() {
            document.getElementById('modal').classList.remove('show');
        }

        function toggleFavorite(btn) {
            const card = btn.closest('.card');
            const index = parseInt(card.dataset.index);
            const params = JSON.parse(card.dataset.params);

            btn.classList.toggle('active');
            card.classList.toggle('favorite');

            if (btn.classList.contains('active')) {
                favorites.push({index: index, params: params});
            } else {
                favorites = favorites.filter(f => f.index !== index);
            }

            updateFavoritesDisplay();
        }

        function updateFavoritesDisplay() {
            document.getElementById('favBtn').textContent = `Show Favorites (${favorites.length})`;
            document.getElementById('favoritesJson').textContent = JSON.stringify(favorites, null, 2);
        }

        function showFavorites() {
            document.getElementById('favoritesList').classList.toggle('show');
        }

        function copyFavorites() {
            navigator.clipboard.writeText(JSON.stringify(favorites, null, 2));
            alert('Favorites copied to clipboard!');
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') hideModal();
        });
    </script>
</body>
</html>
"""

    gallery_path = output_path / "gallery.html"
    with open(gallery_path, "w") as f:
        f.write(html)

    print(f"\nGallery saved to: {gallery_path}")
    return gallery_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate random GLIC variations for exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 20 variations with random presets
    python random_exploration.py input.png output/

    # Generate 50 variations
    python random_exploration.py input.png output/ --count 50

    # Use only specific presets
    python random_exploration.py input.png output/ --presets vhs acid datamosh heavy_glitch

    # Full random parameters (more variety)
    python random_exploration.py input.png output/ --full-random

    # Generate with HTML gallery
    python random_exploration.py input.png output/ --gallery

    # Reproducible generation with seed
    python random_exploration.py input.png output/ --seed 12345

Output:
    - Numbered PNG files (001_preset.png, 002_preset.png, ...)
    - 000_original.png (original image for reference)
    - results.json (all parameters for reproducibility)
    - gallery.html (interactive comparison gallery, with --gallery)
        """
    )

    parser.add_argument("input", help="Input image file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--count", "-n", type=int, default=20, help="Number of variations")
    parser.add_argument("--presets", "-p", nargs="+", help="Specific presets to use")
    parser.add_argument("--full-random", "-r", action="store_true", help="Use fully random parameters")
    parser.add_argument("--seed", "-s", type=int, help="Base seed for reproducibility")
    parser.add_argument("--gallery", "-g", action="store_true", help="Generate HTML gallery")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        results = generate_variations(
            args.input,
            args.output,
            count=args.count,
            presets=args.presets,
            full_random=args.full_random,
            base_seed=args.seed,
            show_progress=not args.quiet
        )

        # Save results JSON
        output_path = Path(args.output)
        results_path = output_path / "results.json"
        with open(results_path, "w") as f:
            json.dump({
                "input": args.input,
                "count": args.count,
                "full_random": args.full_random,
                "seed": args.seed,
                "results": results
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Generate gallery
        if args.gallery:
            generate_html_gallery(args.output, results, args.input)

        # Summary
        success = len([r for r in results if r.get("filename")])
        print(f"\nGenerated {success}/{args.count} variations")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
