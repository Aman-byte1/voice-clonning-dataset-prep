# Voice Cloning Dataset Preparation Pipeline

This repository contains a data pipeline to programmatically generate a multilingual voice cloning dataset using the [Kani-TTS](https://huggingface.co/nineninesix/kani-tts-370m) model. 

It downloads a source dataset of text pairs from HuggingFace, generates high-quality audio for specified speakers and languages, and automatically uploads the resulting audio dataset back to the HuggingFace Hub.

## Setup Instructions

### 1. Clone the Repository

Start by cloning this repository to your server or local machine:

```bash
git clone https://github.com/Aman-byte1/voice-clonning-dataset-prep.git
cd voice-clonning-dataset-prep
```

### 2. Set Your HuggingFace Token

The pipeline requires a HuggingFace Write token to download the source dataset and upload the generated audio dataset. Set it as an environment variable:

**Linux / macOS:**
```bash
export HF_TOKEN="your_hf_token_here"
```

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="your_hf_token_here"
```

### 3. Run the Pipeline

The pipeline is fully automated. You can run it using the provided script. 

**Linux / macOS:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```cmd
run.bat
```

The script will automatically:
1. Install all dependencies from `requirements.txt`
2. Run `generate_dataset.py` to create the audio files locally.
3. Run `push_to_hub.py` to correctly format and upload the dataset to your HuggingFace account.

## Configuration

All customizable settings are located in `config.yaml`.

- **`huggingface.source_dataset`**: The dataset containing your text pairs (default: `amanuelbyte/en-fr-translated-dataset`)
- **`huggingface.output_dataset`**: Where to upload the generated dataset (default: `amanuelbyte/vc-en-fr-dataset`)
- **`speakers`**: The Kani-TTS speakers to use (default: `karim`, `david`, `seulgi`)
- **`sample.size`**: The number of rows to process. **Important:** It is currently set to `10` for testing. Remove this or set it to `null` to process the entire dataset!
