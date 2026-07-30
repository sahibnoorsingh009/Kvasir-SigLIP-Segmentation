from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

from .inference import (
    Service,
    compare_overlay,
    gtmask,
    maskimg,
    overlay,
    scores,
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "examples" / "images"
MASK_DIR = BASE_DIR / "examples" / "masks"
METADATA_PATH = BASE_DIR / "metadata.json"

METADATA = (
    json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if METADATA_PATH.exists()
    else {}
)


def find_file(directory: Path, image_id: str) -> Path | None:
    matches = [
        path
        for path in directory.glob(f"{image_id}.*")
        if path.is_file()
    ]
    return matches[0] if matches else None


def get_choices() -> list[str]:
    choices = []

    for image_id, info in METADATA.items():
        if find_file(IMAGE_DIR, image_id) is not None:
            choices.append(
                f"{info.get('title', image_id)} | {image_id}"
            )

    return choices


def get_example_id(choice: str | None) -> str | None:
    if not choice:
        return None

    return choice.rsplit("|", 1)[-1].strip()


def load_example(choice: str | None):
    image_id = get_example_id(choice)

    if image_id is None:
        return None, None, "No example selected."

    image_path = find_file(IMAGE_DIR, image_id)
    mask_path = find_file(MASK_DIR, image_id)

    if image_path is None:
        return None, None, f"Image not found: `{image_id}`"

    image = Image.open(image_path).convert("RGB")
    mask = (
        Image.open(mask_path).convert("L")
        if mask_path is not None
        else None
    )

    info = METADATA.get(image_id, {})

    description = (
        f"### {info.get('title', image_id)}\n"
        f"{info.get('description', '')}\n\n"
        f"**Image ID:** `{image_id}`"
    )

    return image, mask, description


@lru_cache(maxsize=1)
def get_service() -> Service:
    return Service()


def run_comparison(
    uploaded_image,
    uploaded_mask,
    input_mode,
    selected_example,
):
    if input_mode == "Preloaded example":
        image, mask, _ = load_example(selected_example)

        if image is None:
            raise gr.Error("Select a valid curated example.")

        uploaded_image = image
        uploaded_mask = mask

    if uploaded_image is None:
        raise gr.Error("Upload an image or select an example.")

    service = get_service()

    image_array, resunet_result, siglip_result = service.compare(
        uploaded_image
    )

    ground_truth = gtmask(uploaded_mask, image_array)

    resunet_mask = maskimg(resunet_result.mask)
    siglip_mask = maskimg(siglip_result.mask)

    resunet_overlay = overlay(
        image_array,
        resunet_result.mask,
        (0, 145, 255),
    )

    siglip_overlay = overlay(
        image_array,
        siglip_result.mask,
        (0, 220, 80),
    )

    direct_comparison = compare_overlay(
        image_array,
        resunet_result.mask,
        siglip_result.mask,
    )

    table_rows = []

    if ground_truth is not None:
        resunet_scores = scores(
            resunet_result.mask,
            ground_truth,
        )

        siglip_scores = scores(
            siglip_result.mask,
            ground_truth,
        )

        for model_name, result, model_scores in [
            ("ResUNet", resunet_result, resunet_scores),
            ("SigLIP2 Full", siglip_result, siglip_scores),
        ]:
            table_rows.append(
                {
                    "Model": model_name,
                    "Dice": round(float(model_scores["Dice"]), 4),
                    "IoU": round(float(model_scores["IoU"]), 4),
                    "Precision": round(
                        float(model_scores["Precision"]),
                        4,
                    ),
                    "Recall": round(
                        float(model_scores["Recall"]),
                        4,
                    ),
                    "Specificity": round(
                        float(model_scores["Specificity"]),
                        4,
                    ),
                    "Time (s)": round(result.seconds, 4),
                    "Predicted area (%)": round(
                        float(result.mask.mean() * 100),
                        2,
                    ),
                }
            )

        ground_truth_image = maskimg(ground_truth)

        status = (
            "Metrics were calculated using the supplied ground truth. "
            "**Blue contour = ResUNet. Green contour = SigLIP2 Full.**"
        )

    else:
        for model_name, result in [
            ("ResUNet", resunet_result),
            ("SigLIP2 Full", siglip_result),
        ]:
            table_rows.append(
                {
                    "Model": model_name,
                    "Dice": None,
                    "IoU": None,
                    "Precision": None,
                    "Recall": None,
                    "Specificity": None,
                    "Time (s)": round(result.seconds, 4),
                    "Predicted area (%)": round(
                        float(result.mask.mean() * 100),
                        2,
                    ),
                }
            )

        ground_truth_image = None

        status = (
            "No ground-truth mask was supplied, so Dice and IoU "
            "cannot be calculated. "
            "**Blue contour = ResUNet. Green contour = SigLIP2 Full.**"
        )

    return (
        image_array,
        ground_truth_image,
        resunet_mask,
        siglip_mask,
        resunet_overlay,
        siglip_overlay,
        direct_comparison,
        pd.DataFrame(table_rows),
        status,
    )


CSS = """
.gradio-container {
    max-width: 1450px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

body {
    background: #07101d !important;
}

.hero {
    padding: 28px 32px;
    border-radius: 22px;
    margin-bottom: 16px;
    background: linear-gradient(
        135deg,
        #13355d 0%,
        #17608f 100%
    );
    box-shadow: 0 16px 35px rgba(0, 0, 0, 0.25);
}

.hero h1 {
    margin: 0;
    color: white;
    font-size: 2rem;
}

.hero p {
    margin: 8px 0 0;
    color: #dcecff;
    font-size: 1rem;
}

.notice {
    padding: 12px 16px;
    margin-bottom: 18px;
    border: 1px solid #d3ac39;
    border-radius: 12px;
    background: #fff7d7;
    color: #5c4700;
}

.control-panel {
    padding: 18px;
    border: 1px solid #29384f;
    border-radius: 16px;
    background: #111b2b;
}

.section-title {
    margin-top: 14px;
    margin-bottom: 8px;
    font-size: 1.25rem;
    font-weight: 700;
}

.image-card {
    border: 1px solid #29384f;
    border-radius: 14px;
    overflow: hidden;
    background: #050a11;
}

.image-card img {
    object-fit: contain !important;
    background: #050a11 !important;
}

button.primary {
    min-height: 50px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}

footer {
    display: none !important;
}
"""


choices = get_choices()
default_choice = choices[0] if choices else None

with gr.Blocks(
    title="Kvasir-SEG Live Segmentation Comparison"
) as demo:
    gr.HTML(
        """
        <div class="hero">
            <h1>Kvasir-SEG Live Segmentation Comparison</h1>
            <p>
                Compare ResUNet and fully fine-tuned SigLIP2
                on polyp segmentation.
            </p>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="notice">
            <strong>Research demonstration only.</strong>
            This application is not validated for clinical diagnosis,
            treatment, or patient management.
            Do not upload identifiable patient information.
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes=["control-panel"]):
                gr.Markdown("## 1. Select input")

                input_mode = gr.Radio(
                    choices=[
                        "Preloaded example",
                        "Upload custom image",
                    ],
                    value="Preloaded example",
                    label="Input mode",
                )

                selected_example = gr.Dropdown(
                    choices=choices,
                    value=default_choice,
                    label="Curated example",
                )

                example_info = gr.Markdown()

                with gr.Accordion(
                    "Custom image upload",
                    open=False,
                ):
                    image_input = gr.Image(
                        type="pil",
                        label="Endoscopy image",
                        height=260,
                    )

                    mask_input = gr.Image(
                        type="pil",
                        image_mode="L",
                        label="Optional ground-truth mask",
                        height=220,
                    )

                run_button = gr.Button(
                    "Run model comparison",
                    variant="primary",
                    size="lg",
                )

                gr.ClearButton(
                    components=[
                        image_input,
                        mask_input,
                    ],
                    value="Clear uploads",
                )

        with gr.Column(scale=8):
            gr.Markdown("## 2. Input and reference")

            with gr.Row(equal_height=True):
                original_output = gr.Image(
                    label="Original image",
                    height=330,
                    elem_classes=["image-card"],
                )

                ground_truth_output = gr.Image(
                    label="Ground-truth mask",
                    height=330,
                    elem_classes=["image-card"],
                )

    gr.Markdown("## 3. Predicted masks")

    with gr.Row(equal_height=True):
        resunet_mask_output = gr.Image(
            label="ResUNet mask",
            height=340,
            elem_classes=["image-card"],
        )

        siglip_mask_output = gr.Image(
            label="SigLIP2 Full mask",
            height=340,
            elem_classes=["image-card"],
        )

    gr.Markdown("## 4. Prediction overlays")

    with gr.Row(equal_height=True):
        resunet_overlay_output = gr.Image(
            label="ResUNet overlay",
            height=340,
            elem_classes=["image-card"],
        )

        siglip_overlay_output = gr.Image(
            label="SigLIP2 overlay",
            height=340,
            elem_classes=["image-card"],
        )

    gr.Markdown("## 5. Direct contour comparison")

    comparison_output = gr.Image(
        label="Blue: ResUNet | Green: SigLIP2 Full",
        height=500,
        elem_classes=["image-card"],
    )

    gr.Markdown("## 6. Quantitative results")

    metrics_output = gr.Dataframe(
        label="Metrics",
        interactive=False,
        wrap=True,
    )

    status_output = gr.Markdown()

    selected_example.change(
        fn=load_example,
        inputs=[selected_example],
        outputs=[
            image_input,
            mask_input,
            example_info,
        ],
    )

    demo.load(
        fn=load_example,
        inputs=[selected_example],
        outputs=[
            image_input,
            mask_input,
            example_info,
        ],
    )

    run_button.click(
        fn=run_comparison,
        inputs=[
            image_input,
            mask_input,
            input_mode,
            selected_example,
        ],
        outputs=[
            original_output,
            ground_truth_output,
            resunet_mask_output,
            siglip_mask_output,
            resunet_overlay_output,
            siglip_overlay_output,
            comparison_output,
            metrics_output,
            status_output,
        ],
        api_name="compare_models",
    )
