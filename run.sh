#!/bin/bash
# ============================================================
#  Voice Cloning Dataset Pipeline - Run Script (Linux/Mac)
# ============================================================

# Auto-detect the correct Python binary
# Try python3.13 first (RunPod default), then python3, then python
if command -v python3.13 &> /dev/null; then
    PY=python3.13
elif command -v python3 &> /dev/null && python3 -m pip --version &> /dev/null; then
    PY=python3
elif command -v python &> /dev/null && python -m pip --version &> /dev/null; then
    PY=python
else
    echo "[ERROR] No suitable Python with pip found!"
    exit 1
fi

echo "Using Python: $PY ($($PY --version))"

# Ensure HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    export HF_TOKEN="your_token_here"
fi

echo ""
echo "============================================================"
echo " Voice Cloning Dataset Pipeline"
echo "============================================================"
echo ""

echo "[1/4] Installing nemo-toolkit with all its dependencies..."
# Force reinstall so deps aren't skipped if nemo was previously installed with --no-deps
$PY -m pip install --force-reinstall "nemo-toolkit[all]"
if [ $? -ne 0 ]; then
    echo "[WARN] nemo-toolkit install reported errors (likely transformers conflict) - continuing..."
fi

echo "[2/4] Installing kani-tts (no-deps to avoid pinning nemo version)..."
$PY -m pip install kani-tts --no-deps
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install kani-tts"
    exit 1
fi

echo "[3/4] Forcing transformers==4.57.1 (required by LFM2)..."
$PY -m pip install -U "transformers==4.57.1"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install transformers"
    exit 1
fi

echo "[4/4] Installing remaining utilities and missing dependencies..."
$PY -m pip install datasets soundfile PyYAML huggingface_hub numpy hf_transfer matplotlib nv-one-logger lightning pytorch-lightning omegaconf hydra-core einops sentencepiece tensorboard torchmetrics pydantic lhotse editdistance librosa inflect unidecode tqdm scipy pydub
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install base dependencies"
    exit 1
fi

echo ""
echo "[5/5] Generating audio dataset..."
$PY generate_dataset.py --config config.yaml
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to generate dataset"
    exit 1
fi

echo ""
echo "[6/6] Pushing dataset to HuggingFace..."
$PY push_to_hub.py --config config.yaml
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to push dataset"
    exit 1
fi

echo ""
echo "============================================================"
echo " Pipeline completed successfully!"
echo "============================================================"
