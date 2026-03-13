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

# Step 1: Install kani-tts without its strict dependencies first
echo "[1/4] Installing kani-tts (no-deps to avoid conflicts)..."
$PY -m pip install kani-tts --no-deps
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install kani-tts"
    exit 1
fi

echo "[2/4] Installing base dependencies..."
$PY -m pip install datasets soundfile PyYAML huggingface_hub numpy hf_transfer torch librosa scipy omegaconf onnx protobuf ruamel.yaml scikit-learn tensorboard text-unidecode wget wrapt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install base dependencies"
    exit 1
fi

echo "[3/4] Installing transformers==4.57.1 for LFM2..."
$PY -m pip install -U "transformers==4.57.1"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install transformers"
    exit 1
fi

echo "[4/4] Installing nemo-toolkit (no-deps to avoid conflicts)..."
$PY -m pip install "nemo-toolkit[all]" --no-deps

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
