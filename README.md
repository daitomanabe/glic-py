# GLIC - GLitch Image Codec

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of GLIC - a sophisticated image codec that intentionally creates artistic glitch effects.

![GLIC Example](https://via.placeholder.com/800x300?text=GLIC+Glitch+Art)

## Features

- **20+ Artistic Presets**: vhs, datamosh, acid, noise, dreamy, and more
- **16 Color Spaces**: RGB, HSB, HWB, YUV, LAB, and more
- **24 Prediction Methods**: SPIRAL, WAVE, GRADIENT, RADIAL, NOISE, etc.
- **41 Wavelet Types**: Haar, Daubechies, Symlet, Coiflet, Biorthogonal
- **6 Post-Processing Effects**: Scanline, Chromatic, Pixelate, Dither, Posterize, Glitch
- **Web GUI**: Browser-based interface with Gradio
- **Batch Processing**: Process entire folders with parallel execution
- **Simple API**: One-liner functions for quick use

## Installation

### From PyPI (recommended)
```bash
pip install glic
```

### With GUI support
```bash
pip install glic[gui]
```

### From source
```bash
git clone https://github.com/daito-manabe/glic-py.git
cd glic-py
pip install -e .

# With all options
pip install -e ".[all]"
```

## Quick Start

### One-Liner API

```python
import glic

# Basic glitch
glic.glitch("input.png", "output.png")

# With preset
glic.glitch("input.png", "output.png", preset="heavy_glitch")

# Custom intensity
glic.glitch("input.png", "output.png", intensity=0.8)

# With effects
glic.glitch("input.png", "output.png", effects=["scanline", "chromatic"])

# Random glitch
glic.random_glitch("input.png", "output.png")

# Create variations
glic.create_variations("input.png", "output_folder/", num_variations=5)

# Batch processing
glic.batch_glitch("input_folder/", "output_folder/")
```

### Command Line

```bash
# Apply glitch effect
glic glitch input.png output.png

# Use a preset
glic glitch input.png output.png --preset heavy_glitch

# Custom parameters
glic glitch input.png output.png \
    --colorspace YUV \
    --prediction SPIRAL \
    --quantization 150

# Add effects
glic glitch input.png output.png \
    --effect scanline \
    --effect chromatic

# List presets
glic --list-presets
```

### Web GUI

```bash
# Install with GUI support
pip install glic[gui]

# Launch web interface
python -m glic.gui

# Or with Makefile
make gui
```

Open `http://localhost:7860` in your browser.

## Available Presets

| Preset | Description |
|--------|-------------|
| `default` | Standard balanced glitch |
| `heavy_glitch` | Strong glitch effects |
| `minimal` | Subtle glitch |
| `high_quality` | Minimal distortion |
| `vhs` | VHS/Retro TV look |
| `datamosh` | Datamosh-style compression artifacts |
| `acid` | Psychedelic color effects |
| `noise` | Static noise pattern |
| `dreamy` | Soft, dreamy distortion |
| `blocks` | Blocky glitch pattern |
| `corrupted` | Heavy data corruption look |
| `mosaic` | Mosaic tile effect |
| `edges` | Edge detection style |
| `interference` | Interference pattern |
| `analog` | Analog TV signal |
| `displacement` | Pixel displacement |
| `color_waves` | Wave-based color distortion |
| `pixelated` | Blocky pixel effect |
| `spiral` | Spiral pattern distortion |
| `chromatic` | Channel separation |

## Advanced Usage

### Custom Configuration

```python
from glic import GlicCodec, CodecConfig, ColorSpace, PredictionMethod

config = CodecConfig()
config.color_space = ColorSpace.YUV

for ch in config.channels:
    ch.prediction_method = PredictionMethod.SPIRAL
    ch.quantization_value = 150

codec = GlicCodec(config)
codec.encode_decode("input.png", "output.png")
```

### Working with NumPy Arrays

```python
import numpy as np
from PIL import Image
from glic import glitch_image

# Load image
img = np.array(Image.open("input.png"))

# Apply glitch
result = glitch_image(img, intensity=0.7, preset="acid")

# Save
Image.fromarray(result).save("output.png")
```

### Adding Effects

```python
from glic import GlicCodec, EffectType
from glic.effects import create_effect_config

codec = GlicCodec()

# Add scanline effect
codec.add_effect(create_effect_config(
    EffectType.SCANLINE,
    intensity=70
))

# Add chromatic aberration
codec.add_effect(create_effect_config(
    EffectType.CHROMATIC,
    intensity=50
))

codec.encode_decode("input.png", "output.png")
```

### Batch Processing with Progress

```python
import glic

# Process all images in a folder
results = glic.batch_glitch(
    "input_folder/",
    "output_folder/",
    preset="vhs",
    intensity=0.6,
    max_workers=8,  # Parallel processing
    progress=True   # Show progress
)
```

## Parameters

### Color Spaces
`RGB`, `HSB`, `HWB`, `OHTA`, `CMY`, `XYZ`, `YXY`, `LAB`, `LUV`, `HCL`, `YUV`, `YPbPr`, `YCbCr`, `YDbDr`, `GS`, `RGGBG`

### Prediction Methods
`PAETH`, `SPIRAL`, `WAVE`, `GRADIENT`, `RADIAL`, `NOISE`, `CHECKERBOARD`, `EDGE`, `MIRROR`, `TRUEMOTION`, `MEDIAN`, `DC`, `CORNER`, `H`, `V`, and more

### Post-Processing Effects
| Effect | Description |
|--------|-------------|
| `scanline` | CRT monitor lines |
| `chromatic` | RGB channel separation |
| `pixelate` | Mosaic effect |
| `dither` | Bayer dithering |
| `posterize` | Color reduction |
| `glitch_shift` | Row displacement |

## Development

```bash
# Clone
git clone https://github.com/daito-manabe/glic-py.git
cd glic-py

# Install dev dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Lint
make lint

# Build package
make build
```

## Project Structure

```
glic-py/
├── glic/                # Main package
│   ├── __init__.py      # Easy API exports
│   ├── easy.py          # Simple one-liner functions
│   ├── codec.py         # GlicCodec class
│   ├── config.py        # Configuration classes
│   ├── colorspaces.py   # 16 color conversions
│   ├── prediction.py    # 24 prediction methods
│   ├── wavelet.py       # 41 wavelet types
│   ├── encoding.py      # 6 encoding methods
│   ├── effects.py       # 6 post-effects
│   ├── segment.py       # Quad-tree segmentation
│   ├── planes.py        # Image management
│   ├── bitio.py         # Bit-level I/O
│   ├── preset_loader.py # Preset system
│   ├── gui.py           # Web GUI (Gradio)
│   └── __main__.py      # CLI
├── examples/            # Example scripts
├── pyproject.toml       # Package config
├── Makefile             # Dev tasks
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE)

## Credits

Python port based on the original GLIC C++ implementation.

Original concept: GlitchCodec/GLIC (Java/Processing)
