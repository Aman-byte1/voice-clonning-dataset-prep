#!/bin/bash
# ============================================================
#  Voice Cloning Dataset Pipeline - Run Script (Linux/Mac)
# ============================================================

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
echo "[1/4] Installing kani-tts and allowing resolution overrides..."
pip install kani-tts --no-deps
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install kani-tts"
    exit 1
fi

echo "[2/4] Installing base dependencies..."
pip install datasets soundfile PyYAML huggingface_hub numpy
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install base dependencies"
    exit 1
fi

echo "[3/4] Forcing upgrade of transformers for LFM2..."
pip install -U "transformers==4.57.1"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to upgrade transformers"
    exit 1
fi

echo "[4/4] Installing remaining nemo-toolkit dependencies (ignoring transformers conflict)..."
pip install "nemo-toolkit[all]" --no-deps

echo ""
echo "[2/3] Generating audio dataset..."
python generate_dataset.py --config config.yaml
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to generate dataset"
    exit 1
fi

echo ""
echo "[3/3] Pushing dataset to HuggingFace..."
python push_to_hub.py --config config.yaml
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to push dataset"
    exit 1
fi

echo ""
echo "============================================================"
echo " Pipeline completed successfully!"
echo "============================================================"
