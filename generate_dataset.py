"""
Voice Cloning Dataset Generator
================================
Downloads text pairs from HuggingFace, generates audio using kani-tts
for each speaker × language combination, and saves a structured dataset.
"""

import os
import sys
import json
import yaml
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime


def load_config(config_path="config.yaml"):
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_huggingface(config):
    """Set up HuggingFace authentication."""
    token = os.environ.get(config["huggingface"]["token_env"])
    if not token:
        print(f"[ERROR] HuggingFace token not found in environment variable "
              f"'{config['huggingface']['token_env']}'")
        print("Please run: export HF_TOKEN=your_token_here")
        sys.exit(1)
    
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print("[OK] HuggingFace authentication successful")
    return token


def load_source_dataset(config):
    """Load and optionally sample the source translation dataset."""
    from datasets import load_dataset

    dataset_name = config["huggingface"]["source_dataset"]
    print(f"[INFO] Loading dataset: {dataset_name}")
    
    ds = load_dataset(dataset_name, split="train")
    print(f"[INFO] Full dataset has {len(ds)} rows")

    sample_size = config["sample"]["size"]
    seed = config["sample"]["seed"]

    if sample_size and sample_size < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample_size))
        print(f"[INFO] Sampled {sample_size} rows (seed={seed})")
    
    return ds


def init_tts_model(config):
    """Initialize the Kani TTS model."""
    from kani_tts import KaniTTS

    model_name = config["model"]["name"]
    print(f"[INFO] Loading TTS model: {model_name}")
    model = KaniTTS(model_name)
    print("[OK] TTS model loaded successfully")

    # Show available speakers
    print("[INFO] Available speakers:")
    model.show_speakers()
    
    return model


def generate_audio(model, text, speaker_id, config):
    """Generate audio for a single text using the given speaker."""
    try:
        audio, _ = model(text, speaker_id=speaker_id)
        return audio
    except Exception as e:
        print(f"[WARN] Failed to generate audio for speaker={speaker_id}: {e}")
        return None


def save_audio(audio, filepath, sample_rate):
    """Save audio array to a WAV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, audio, sample_rate)


def run_pipeline(config_path="config.yaml"):
    """Main pipeline: download text → generate audio → save files."""
    
    # 1. Load config
    config = load_config(config_path)
    print("=" * 60)
    print("  Voice Cloning Dataset Generator")
    print("=" * 60)
    print(f"[INFO] Started at {datetime.now().isoformat()}")
    
    # 2. Setup HuggingFace
    setup_huggingface(config)
    
    # 3. Load source dataset
    ds = load_source_dataset(config)
    
    # 4. Initialize TTS model
    model = init_tts_model(config)
    
    # 5. Prepare output directory
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_format = config["output"]["audio_format"]
    sample_rate = config["model"]["sample_rate"]
    speakers = config["speakers"]
    languages = config["languages"]
    
    # 6. Generate audio for each row × speaker × language
    manifest = []
    total = len(ds) * len(speakers) * len(languages)
    count = 0
    errors = 0
    
    print(f"\n[INFO] Generating {total} audio files "
          f"({len(ds)} texts × {len(speakers)} speakers × {len(languages)} languages)")
    print("-" * 60)
    
    for row_idx, row in enumerate(ds):
        for speaker in speakers:
            for lang_code, col_name in languages.items():
                count += 1
                text = row[col_name]
                
                # Skip empty text
                if not text or not text.strip():
                    print(f"[{count}/{total}] SKIP empty text "
                          f"(row={row_idx}, speaker={speaker}, lang={lang_code})")
                    continue
                
                # Truncate display text
                display_text = text[:60] + "..." if len(text) > 60 else text
                print(f"[{count}/{total}] speaker={speaker} lang={lang_code} "
                      f"text=\"{display_text}\"")
                
                # Generate audio
                audio = generate_audio(model, text, speaker, config)
                
                if audio is None:
                    errors += 1
                    continue
                
                # Save audio file
                filename = f"row{row_idx:05d}_{speaker}_{lang_code}.{audio_format}"
                rel_path = os.path.join(speaker, lang_code, filename)
                abs_path = output_dir / rel_path
                save_audio(audio, str(abs_path), sample_rate)
                
                # Add to manifest
                manifest.append({
                    "row_index": row_idx,
                    "text": text,
                    "speaker_id": speaker,
                    "language": lang_code,
                    "audio_path": rel_path,
                    "sample_rate": sample_rate,
                })
    
    # 7. Save manifest
    manifest_path = output_dir / config["output"]["manifest_file"]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print("-" * 60)
    print(f"[DONE] Generated {len(manifest)} audio files ({errors} errors)")
    print(f"[DONE] Manifest saved to: {manifest_path}")
    print(f"[DONE] Audio files saved to: {output_dir}")
    print(f"[INFO] Finished at {datetime.now().isoformat()}")
    
    return manifest


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Cloning Dataset Generator")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    args = parser.parse_args()
    
    run_pipeline(args.config)
