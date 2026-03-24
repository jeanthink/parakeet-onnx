# parakeet-onnx

NVIDIA Parakeet TDT → ONNX conversion pipeline for [VoiceMate](https://github.com/jeanthink/VoiceMate) Windows.

## What This Does

This repo converts NVIDIA's Parakeet TDT speech recognition models from their native NeMo/PyTorch format into ONNX, so VoiceMate's Windows app can run them locally via ONNX Runtime — no Python, no CUDA, no cloud API needed.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Conversion Pipeline                       │
│                                                              │
│  HuggingFace ──► NeMo Checkpoint ──► ONNX Export ──► Release │
│                                                              │
│  nvidia/parakeet-tdt-0.6b-v2                                 │
│  nvidia/parakeet-tdt-0.6b-v3                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  ONNX Model Components                       │
│                                                              │
│  Audio ──► Preprocessor ──► Encoder ──► Decoder + Joint      │
│  (WAV)     (Mel specs)      (FastConformer)  (TDT)           │
│                                                              │
│  preprocessor.onnx  encoder.onnx  decoder.onnx  joint.onnx  │
│                                   vocabulary.json             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                VoiceMate Windows (.NET)                       │
│                                                              │
│  Downloads ONNX from GitHub Releases                         │
│  Runs inference via Microsoft.ML.OnnxRuntime                 │
│  Fully local — no cloud, no Python at runtime                │
└─────────────────────────────────────────────────────────────┘
```

## Available Models

| Version | HuggingFace Model | Language | Size (approx) |
|---------|-------------------|----------|---------------|
| v2 | `nvidia/parakeet-tdt-0.6b-v2` | English | ~600 MB |
| v3 | `nvidia/parakeet-tdt-0.6b-v3` | Multilingual (25 EU languages) | ~600 MB |

## Run Locally

### Prerequisites

- Python 3.10+
- ~4 GB free disk space (model download + ONNX export)
- ~8 GB RAM recommended
- **HuggingFace Account** with model access (see [Authentication](#authentication) below)

### Authentication

NVIDIA's Parakeet models on HuggingFace are gated (require license acceptance). To download them:

1. **Create a HuggingFace account**: https://huggingface.co/join
2. **Accept the license**:
   - Visit https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
   - Click "Access repository" and accept the license terms
   - Repeat for v3: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
3. **Create a HuggingFace token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with `read` permission
   - Copy the token
4. **Set the HF_TOKEN environment variable**:
   ```bash
   # Linux/macOS
   export HF_TOKEN="hf_your_token_here"
   
   # Windows
   set HF_TOKEN=hf_your_token_here
   # or add to .env file in repo root (will be auto-loaded)
   ```

### Steps

```bash
# Clone the repo
git clone https://github.com/jeanthink/parakeet-onnx.git
cd parakeet-onnx

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Ensure HF_TOKEN is set (see Authentication section above)
export HF_TOKEN="hf_your_token_here"

# Convert v2 model (English)
python convert.py --version v2

# Convert v3 model (Multilingual: 25 European languages)
python convert.py --version v3

# List available models
python convert.py --list-models
```

Output artifacts are written to `output/`:

```
output/
├── preprocessor.onnx    # Audio → mel spectrogram features
├── encoder.onnx         # FastConformer encoder (~500 MB)
├── decoder.onnx         # TDT prediction network
├── joint.onnx           # Joint decision network
├── vocabulary.json      # SentencePiece token ↔ ID mapping
└── config.json          # Model metadata (sample rate, vocab size)
```

## How VoiceMate Uses This

1. **GitHub Actions** runs `convert.py` on schedule or manual trigger
2. ONNX files are uploaded as **GitHub Release** artifacts
3. VoiceMate Windows downloads the ONNX files on first use (via `ModelDownloadManager`)
4. At runtime, VoiceMate loads each ONNX component into **ONNX Runtime** and orchestrates the inference pipeline:
   - Audio capture → Preprocessor → Encoder → Decoder + Joint → Text

The user never sees Python or PyTorch — they get a native Windows experience with local, offline speech recognition.

## GitHub Actions

The workflow (`.github/workflows/convert.yml`) supports:

- **Manual dispatch**: Choose model version (v2/v3) from the Actions UI
- **Monthly schedule**: Automatically checks for model updates on the 1st of each month
- **Release creation**: Each run creates a tagged GitHub Release with all ONNX artifacts

### Setup

For the workflow to download models, you must add the `HF_TOKEN` secret to your repo:

1. Follow the [Authentication](#authentication) section above to create a HuggingFace token
2. Go to **Settings → Secrets and variables → Actions** in your GitHub repo
3. Click **New repository secret**
4. Name: `HF_TOKEN`, Value: (paste your HuggingFace token)
5. Save

The workflow will now be able to download and convert the models.

## License

MIT — see [LICENSE](LICENSE).
