"""
Push Generated Dataset to HuggingFace Hub
==========================================
Reads the generated audio files + manifest, builds a HuggingFace Dataset
with Audio features, and pushes it to the Hub.
"""

import os
import sys
import json
import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def push_dataset(config_path="config.yaml"):
    """Build and push the audio dataset to HuggingFace Hub."""
    from datasets import Dataset, Audio, Features, Value
    from huggingface_hub import login

    # Load config
    config = load_config(config_path)
    
    # Authenticate
    token = os.environ.get(config["huggingface"]["token_env"])
    if not token:
        print(f"[ERROR] HuggingFace token not found in environment variable "
              f"'{config['huggingface']['token_env']}'")
        sys.exit(1)
    
    login(token=token, add_to_git_credential=False)
    print("[OK] HuggingFace authentication successful")
    
    # Load manifest
    output_dir = Path(config["output"]["directory"])
    manifest_path = output_dir / config["output"]["manifest_file"]
    
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at {manifest_path}")
        print("Run generate_dataset.py first to create the dataset.")
        sys.exit(1)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    print(f"[INFO] Loaded manifest with {len(manifest)} entries")
    
    # Build dataset dict with absolute audio paths
    dataset_dict = {
        "text": [],
        "audio": [],
        "speaker_id": [],
        "language": [],
    }
    
    skipped = 0
    for entry in manifest:
        audio_abs_path = str(output_dir / entry["audio_path"])
        
        if not os.path.exists(audio_abs_path):
            print(f"[WARN] Audio file not found: {audio_abs_path}")
            skipped += 1
            continue
        
        dataset_dict["text"].append(entry["text"])
        dataset_dict["audio"].append(audio_abs_path)
        dataset_dict["speaker_id"].append(entry["speaker_id"])
        dataset_dict["language"].append(entry["language"])
    
    if skipped > 0:
        print(f"[WARN] Skipped {skipped} entries with missing audio files")
    
    # Create HuggingFace Dataset with Audio feature
    features = Features({
        "text": Value("string"),
        "audio": Audio(sampling_rate=config["model"]["sample_rate"]),
        "speaker_id": Value("string"),
        "language": Value("string"),
    })
    
    ds = Dataset.from_dict(dataset_dict)
    ds = ds.cast(features)
    
    print(f"[INFO] Dataset created with {len(ds)} rows")
    print(f"[INFO] Features: {ds.features}")
    
    # Push to Hub
    output_dataset = config["huggingface"]["output_dataset"]
    print(f"[INFO] Pushing to: {output_dataset}")
    
    ds.push_to_hub(
        output_dataset,
        token=token,
        private=False,
    )
    
    print(f"[DONE] Dataset pushed to https://huggingface.co/datasets/{output_dataset}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Push voice dataset to HuggingFace")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    args = parser.parse_args()
    
    push_dataset(args.config)
