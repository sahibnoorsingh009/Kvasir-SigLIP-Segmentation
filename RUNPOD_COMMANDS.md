# RunPod setup
Recommended: 1 x RTX 4090 24 GB or better, PyTorch 2.8, CUDA 12.8, 50 GB volume, expose HTTP port 7860.

```bash
cd /workspace
git clone https://github.com/sahibnoorsingh009/Kvasir-SigLIP-Segmentation.git
cd Kvasir-SigLIP-Segmentation
# Copy this package into this repository root
chmod +x scripts/setup_runpod.sh
bash scripts/setup_runpod.sh
python scripts/verify_demo.py
python app.py
```
Background mode:
```bash
nohup python app.py > demo.log 2>&1 &
tail -f demo.log
```
