# Kvasir-SigLIP Live Demo Add-on

This package adds a complete Gradio application to the existing project:
https://github.com/sahibnoorsingh009/Kvasir-SigLIP-Segmentation

## Features
- Upload a custom endoscopy image
- Optional ground-truth mask
- Select six curated Kvasir-SEG examples
- ResUNet and SigLIP2 Full masks
- Individual overlays and contour comparison
- Dice, IoU, precision, recall and specificity when a mask is available
- Automatic checkpoint download from `Sahibnoor1/kvasir-siglip2-segmentation-checkpoints`
- Automatic curated-example preparation via KaggleHub

## Installation
Extract/copy these files into the root of the main GitHub repository, then run:
```bash
bash scripts/setup_runpod.sh
python app.py
```
Open port 7860.

## Hugging Face Space
Copy `src`, `configs`, `demo`, `app.py`, and `requirements-demo.txt` into a GPU Gradio Space. Rename `requirements-demo.txt` to `requirements.txt` and use `README_SPACE.md` as the Space `README.md`.

## GitHub Pages
Edit `docs/index.html`, replacing `YOUR_SPACE_URL` with the deployed Space subdomain, then enable GitHub Pages from the `docs/` folder.

Research demonstration only. Not validated for clinical use.
