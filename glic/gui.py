#!/usr/bin/env python3
"""
GLIC Web GUI using Gradio.

Run with:
    python -m glic.gui

Or:
    glic-gui (after pip install)
"""

import tempfile
import os
from pathlib import Path
from typing import Optional, List

import numpy as np

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


def check_gradio():
    """Check if Gradio is installed."""
    if not HAS_GRADIO:
        raise ImportError(
            "Gradio is not installed. Install it with:\n"
            "  pip install glic[gui]\n"
            "or:\n"
            "  pip install gradio"
        )


def process_image(
    image: np.ndarray,
    preset: str,
    intensity: float,
    colorspace: str,
    prediction: str,
    use_scanline: bool,
    use_chromatic: bool,
    use_pixelate: bool,
    use_dither: bool,
    effect_intensity: float,
) -> np.ndarray:
    """Process image with GLIC."""
    from glic import glitch_image

    # Build effects list
    effects = []
    if use_scanline:
        effects.append("scanline")
    if use_chromatic:
        effects.append("chromatic")
    if use_pixelate:
        effects.append("pixelate")
    if use_dither:
        effects.append("dither")

    # Use preset or custom settings
    if preset != "custom":
        return glitch_image(
            image,
            preset=preset,
            intensity=intensity,
            effects=effects if effects else None,
        )
    else:
        return glitch_image(
            image,
            intensity=intensity,
            colorspace=colorspace if colorspace != "Auto" else None,
            prediction=prediction if prediction != "Auto" else None,
            effects=effects if effects else None,
        )


def create_interface():
    """Create Gradio interface."""
    check_gradio()

    from glic import list_presets, list_colorspaces, list_predictions

    presets = ["custom"] + list_presets()
    colorspaces = ["Auto"] + list_colorspaces()
    predictions = ["Auto"] + list_predictions()

    with gr.Blocks(
        title="GLIC - GLitch Image Codec",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # GLIC - GLitch Image Codec

            Create artistic glitch effects on your images. Upload an image and adjust parameters
            to create unique glitch art.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    height=400,
                )

                with gr.Accordion("Basic Settings", open=True):
                    preset = gr.Dropdown(
                        choices=presets,
                        value="default",
                        label="Preset",
                        info="Choose a preset or 'custom' for manual control"
                    )
                    intensity = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Glitch Intensity",
                        info="Higher = more glitch effect"
                    )

                with gr.Accordion("Advanced Settings", open=False):
                    colorspace = gr.Dropdown(
                        choices=colorspaces,
                        value="Auto",
                        label="Color Space",
                        info="Color space for processing"
                    )
                    prediction = gr.Dropdown(
                        choices=predictions,
                        value="Auto",
                        label="Prediction Method",
                        info="Algorithm for creating glitch patterns"
                    )

                with gr.Accordion("Post Effects", open=False):
                    use_scanline = gr.Checkbox(
                        label="Scanline (CRT effect)",
                        value=False
                    )
                    use_chromatic = gr.Checkbox(
                        label="Chromatic Aberration",
                        value=False
                    )
                    use_pixelate = gr.Checkbox(
                        label="Pixelate",
                        value=False
                    )
                    use_dither = gr.Checkbox(
                        label="Dither",
                        value=False
                    )
                    effect_intensity = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Effect Intensity"
                    )

                process_btn = gr.Button(
                    "Apply Glitch",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Output Image",
                    type="numpy",
                    height=400,
                )

                download_btn = gr.DownloadButton(
                    label="Download Result",
                    visible=False,
                )

        # Examples
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["default", 0.5],
                ["heavy_glitch", 0.7],
                ["minimal", 0.3],
                ["color_waves", 0.6],
                ["spiral", 0.5],
            ],
            inputs=[preset, intensity],
            label="Click to try preset"
        )

        # Event handlers
        def on_process(img, preset, intensity, colorspace, prediction,
                       scanline, chromatic, pixelate, dither, eff_int):
            if img is None:
                return None

            result = process_image(
                img, preset, intensity, colorspace, prediction,
                scanline, chromatic, pixelate, dither, eff_int
            )
            return result

        process_btn.click(
            fn=on_process,
            inputs=[
                input_image, preset, intensity, colorspace, prediction,
                use_scanline, use_chromatic, use_pixelate, use_dither,
                effect_intensity
            ],
            outputs=output_image
        )

        # Quick preview on parameter change
        for control in [preset, intensity]:
            control.change(
                fn=on_process,
                inputs=[
                    input_image, preset, intensity, colorspace, prediction,
                    use_scanline, use_chromatic, use_pixelate, use_dither,
                    effect_intensity
                ],
                outputs=output_image
            )

        gr.Markdown(
            """
            ---
            **GLIC** - GLitch Image Codec | [GitHub](https://github.com/daito-manabe/glic-py)

            Tips:
            - Try different **presets** for various glitch styles
            - Increase **intensity** for stronger effects
            - Combine **post effects** for unique looks
            - Use **color spaces** like YUV or HSB for different color distortions
            """
        )

    return app


def launch(share: bool = False, port: int = 7860):
    """Launch the Gradio web interface."""
    check_gradio()
    app = create_interface()
    app.launch(share=share, server_port=port)


def main():
    """Main entry point for GUI."""
    import argparse

    parser = argparse.ArgumentParser(description="GLIC Web GUI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )

    args = parser.parse_args()
    launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
