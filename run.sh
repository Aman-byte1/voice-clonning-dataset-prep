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

# Step 1: Install dependencies
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

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
