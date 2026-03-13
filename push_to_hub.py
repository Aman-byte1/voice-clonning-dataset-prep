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
    # Group by (row_index, speaker_id) to create a parallel dataset
    languages = list(config.get("languages", {"en": "", "fr": ""}).keys())
    
    # Initialize the parallel Dictionary
    dataset_dict = { "speaker_id": [] }
    for lang in languages:
        dataset_dict[f"text_{lang}"] = []
        dataset_dict[f"audio_{lang}"] = []
        
    # Group manifest entries
    grouped_entries = {}
    for entry in manifest:
        key = (entry["row_index"], entry["speaker_id"])
        if key not in grouped_entries:
            grouped_entries[key] = {}
            
        lang = entry["language"]
        audio_abs_path = str(output_dir / entry["audio_path"])
        
        if os.path.exists(audio_abs_path):
            grouped_entries[key][lang] = {
                "text": entry["text"],
                "audio": audio_abs_path
            }
            
    skipped = 0
    # Populate the dataset dictionary
    for (row_idx, speaker), lang_data in grouped_entries.items():
        # Check if we have all languages for this parallel row
        has_all_langs = all(lang in lang_data for lang in languages)
        
        if not has_all_langs:
            skipped += 1
            continue
            
        dataset_dict["speaker_id"].append(speaker)
        for lang in languages:
            dataset_dict[f"text_{lang}"].append(lang_data[lang]["text"])
            dataset_dict[f"audio_{lang}"].append(lang_data[lang]["audio"])
    
    if skipped > 0:
        print(f"[WARN] Skipped {skipped} parallel rows due to missing audio files")
    
    # Create HuggingFace Dataset with parallel Audio features
    features_dict = {
        "speaker_id": Value("string")
    }
    for lang in languages:
        features_dict[f"text_{lang}"] = Value("string")
        features_dict[f"audio_{lang}"] = Audio(sampling_rate=config["model"]["sample_rate"])
        
    features = Features(features_dict)
    
    ds = Dataset.from_dict(dataset_dict)
    ds = ds.cast(features)
    
    print(f"[INFO] Parallel Dataset created with {len(ds)} rows")
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
