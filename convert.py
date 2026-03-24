#!/usr/bin/env python3
"""
Parakeet TDT → ONNX Conversion Script

Downloads NVIDIA Parakeet TDT models from HuggingFace and exports each
component (Preprocessor, Encoder, Decoder Joint) to ONNX format for use
with ONNX Runtime in VoiceMate Windows.

Usage:
    python convert.py --version v2
    python convert.py --version v3
    python convert.py --list-models
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import onnx

# Model registry: version → HuggingFace model ID
MODELS = {
    "v2": "nvidia/parakeet-tdt-0.6b-v2",
    "v3": "nvidia/parakeet-tdt-0.6b-v3",
}

OUTPUT_DIR = Path("output")


def list_models():
    """Print available model versions and their HuggingFace repos."""
    print("Available Parakeet TDT models:")
    print()
    for version, hf_id in MODELS.items():
        print(f"  {version}: {hf_id}")
        print(f"       https://huggingface.co/{hf_id}")
        print()


def download_model(version: str):
    """Download NeMo checkpoint from HuggingFace."""
    import nemo.collections.asr as nemo_asr

    hf_id = MODELS[version]
    hf_token = os.environ.get("HF_TOKEN")

    # NeMo uses huggingface_hub internally — set the token via login() if provided
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    print(f"[1/5] Downloading model: {hf_id}")
    print(f"       This may take several minutes on first run...")

    try:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=hf_id,
            map_location="cpu",
        )
        print(f"       Model loaded: {type(model).__name__}")
        return model
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e) or "authentication" in str(e).lower():
            print(f"       ERROR: Authentication required for {hf_id}")
            print(f"       This model requires accepting license terms on HuggingFace.")
            print(f"       Please:")
            print(f"       1. Visit: https://huggingface.co/{hf_id}")
            print(f"       2. Accept the license terms")
            print(f"       3. Create a HuggingFace token: https://huggingface.co/settings/tokens")
            print(f"       4. Set HF_TOKEN environment variable or pass --token to this script")
            sys.exit(1)
        else:
            print(f"       ERROR: Failed to download model: {e}")
            sys.exit(1)


def export_preprocessor(model, export_dir: Path):
    """Export the audio preprocessor (mel spectrogram) to ONNX."""
    import torch
    import torch.onnx

    print("[2/5] Exporting Preprocessor to ONNX...")
    preprocessor = model.preprocessor
    preprocessor.eval()

    # Preprocessor expects raw audio waveform
    dummy_signal = torch.randn(1, 16000, device="cpu")  # 1 second of audio
    dummy_length = torch.tensor([16000], dtype=torch.long, device="cpu")

    preprocessor_path = export_dir / "preprocessor.onnx"

    torch.onnx.export(
        preprocessor,
        (dummy_signal, dummy_length),
        str(preprocessor_path),
        input_names=["audio_signal", "length"],
        output_names=["processed_signal", "processed_length"],
        dynamic_axes={
            "audio_signal": {0: "batch", 1: "time"},
            "length": {0: "batch"},
            "processed_signal": {0: "batch", 2: "time_steps"},
            "processed_length": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"       Saved: {preprocessor_path} ({preprocessor_path.stat().st_size / 1e6:.1f} MB)")


def export_encoder(model, export_dir: Path):
    """Export the encoder (Conformer/FastConformer) to ONNX."""
    import torch
    import torch.onnx

    print("[3/5] Exporting Encoder to ONNX...")
    encoder = model.encoder
    encoder.eval()

    n_mels = model.preprocessor.featurizer.nfilt if hasattr(model.preprocessor, "featurizer") else 80
    dummy_features = torch.randn(1, n_mels, 100, device="cpu")
    dummy_length = torch.tensor([100], dtype=torch.long, device="cpu")

    encoder_path = export_dir / "encoder.onnx"

    torch.onnx.export(
        encoder,
        (dummy_features, dummy_length),
        str(encoder_path),
        input_names=["audio_signal", "length"],
        output_names=["encoded", "encoded_length"],
        dynamic_axes={
            "audio_signal": {0: "batch", 2: "time_steps"},
            "length": {0: "batch"},
            "encoded": {0: "batch", 1: "time_steps"},
            "encoded_length": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"       Saved: {encoder_path} ({encoder_path.stat().st_size / 1e6:.1f} MB)")


def export_decoder_joint(model, export_dir: Path):
    """Export the TDT decoder and joint network to ONNX."""
    import torch
    import torch.onnx

    print("[4/5] Exporting Decoder + Joint to ONNX...")

    # TDT models use an RNNT-style decoder + joint network
    decoder = model.decoder
    joint = model.joint
    decoder.eval()
    joint.eval()

    # Decoder: predicts next token given previous tokens
    vocab_size = model.decoder.vocab_size if hasattr(model.decoder, "vocab_size") else 1024
    pred_len = model.decoder.pred_hidden if hasattr(model.decoder, "pred_hidden") else 640

    # Export decoder prediction network
    dummy_targets = torch.zeros(1, 1, dtype=torch.long, device="cpu")
    dummy_states = None
    if hasattr(decoder, "predict"):
        # NeMo RNNT decoder uses predict() method
        decoder_path = export_dir / "decoder.onnx"
        try:
            g, states = decoder.predict(dummy_targets, state=dummy_states, add_sos=False)
            # Wrap in a module for export
            class DecoderWrapper(torch.nn.Module):
                def __init__(self, dec):
                    super().__init__()
                    self.dec = dec

                def forward(self, targets):
                    g, _ = self.dec.predict(targets, state=None, add_sos=False)
                    return g

            wrapper = DecoderWrapper(decoder)
            torch.onnx.export(
                wrapper,
                (dummy_targets,),
                str(decoder_path),
                input_names=["targets"],
                output_names=["decoder_output"],
                dynamic_axes={
                    "targets": {0: "batch", 1: "time"},
                    "decoder_output": {0: "batch", 1: "time"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
            print(f"       Saved: {decoder_path} ({decoder_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"       Warning: Decoder export failed ({e}), trying alternative approach...")
            # Fall back to scripting the full decoder
            torch.jit.save(torch.jit.script(decoder), str(export_dir / "decoder.pt"))
            print(f"       Saved decoder as TorchScript fallback")

    # Export joint network
    joint_path = export_dir / "joint.onnx"
    try:
        enc_dim = joint.enc_dim if hasattr(joint, "enc_dim") else 512
        pred_dim = joint.pred_dim if hasattr(joint, "pred_dim") else 640
        dummy_enc = torch.randn(1, 1, enc_dim, device="cpu")
        dummy_pred = torch.randn(1, 1, pred_dim, device="cpu")

        class JointWrapper(torch.nn.Module):
            def __init__(self, joint_net):
                super().__init__()
                self.joint = joint_net

            def forward(self, encoder_output, decoder_output):
                return self.joint.joint_after_projection(
                    encoder_output.unsqueeze(2),
                    decoder_output.unsqueeze(1),
                ).squeeze(2).squeeze(1)

        joint_wrapper = JointWrapper(joint)
        torch.onnx.export(
            joint_wrapper,
            (dummy_enc, dummy_pred),
            str(joint_path),
            input_names=["encoder_output", "decoder_output"],
            output_names=["logits"],
            dynamic_axes={
                "encoder_output": {0: "batch"},
                "decoder_output": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"       Saved: {joint_path} ({joint_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"       Warning: Joint export failed ({e}), trying NeMo's built-in export...")
        try:
            model.export(str(export_dir / "model.onnx"))
            print(f"       Saved via NeMo export API")
        except Exception as e2:
            print(f"       NeMo export also failed: {e2}")
            print(f"       Manual export may be required — see README.md")


def export_vocabulary(model, export_dir: Path):
    """Export the tokenizer vocabulary as JSON."""
    print("[5/5] Exporting vocabulary...")

    vocab_path = export_dir / "vocabulary.json"

    vocab = {}
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
        if hasattr(tokenizer, "vocab"):
            vocab = tokenizer.vocab
        elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "id_to_piece"):
            sp = tokenizer.tokenizer
            vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
        else:
            # Try to extract from sentencepiece model file
            try:
                import sentencepiece as spm
                sp_model = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else None
                if sp_model and hasattr(sp_model, "serialized_model_proto"):
                    vocab = {sp_model.id_to_piece(i): i for i in range(sp_model.get_piece_size())}
            except ImportError:
                print("       Warning: sentencepiece not installed, vocab extraction limited")

    if not vocab:
        # Fallback: try decoding range of token IDs
        if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "ids_to_text"):
            vocab = {}
            for i in range(1025):
                try:
                    text = model.tokenizer.ids_to_text([i])
                    vocab[text] = i
                except Exception:
                    break
            print(f"       Extracted {len(vocab)} tokens via ids_to_text")

    # Save vocabulary
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"       Saved: {vocab_path} ({len(vocab)} tokens)")

    # Also save model config metadata
    config = {
        "model_name": type(model).__name__,
        "sample_rate": getattr(model.preprocessor, "_sample_rate", 16000),
        "vocab_size": len(vocab),
    }
    config_path = export_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"       Saved: {config_path}")


def embed_external_weights(export_dir: Path):
    """Post-process ONNX files to embed external weight data.

    NeMo's export() may produce ONNX files with external data tensors
    (separate .data files or individual tensor files). This function
    loads each ONNX file with its external data and re-saves with all
    weights embedded directly in the protobuf, then cleans up the
    external files.
    """
    onnx_files = list(export_dir.glob("*.onnx"))
    if not onnx_files:
        return

    print("[Post] Embedding external weights into ONNX files...")

    for onnx_path in onnx_files:
        try:
            model = onnx.load(str(onnx_path), load_external_data=True)
        except Exception as e:
            print(f"       Skipping {onnx_path.name}: could not load ({e})")
            continue

        has_external = any(
            tensor.HasField("data_location")
            and tensor.data_location == onnx.TensorProto.EXTERNAL
            for tensor in model.graph.initializer
        )
        if not has_external:
            print(f"       {onnx_path.name}: weights already embedded — skipping")
            continue

        # Re-save with all data embedded
        embedded_path = onnx_path.with_suffix(".embedded.onnx")
        try:
            onnx.save_model(model, str(embedded_path), save_as_external_data=False)
            # Replace the original
            embedded_path.replace(onnx_path)
            print(f"       {onnx_path.name}: embedded ({onnx_path.stat().st_size / 1e6:.1f} MB)")
        except ValueError as e:
            # Protobuf has a 2GB limit — if the model is too large, keep external
            if "2GB" in str(e) or "too large" in str(e).lower():
                print(f"       {onnx_path.name}: >2GB, keeping external data")
                if embedded_path.exists():
                    embedded_path.unlink()
            else:
                raise

    # Clean up external data files (tensor files, .data files)
    for f in export_dir.iterdir():
        if f.suffix in (".data", ".weight") or (
            f.is_file() and f.suffix == "" and f.name not in ("Makefile",)
            and not f.name.startswith(".")
            and f.name != "LICENSE"
        ):
            # Check if it looks like an external tensor file (no extension, not a known file)
            try:
                # If file is not JSON, ONNX, or known format, it's likely a tensor file
                if f.suffix == "" and f.stat().st_size > 0:
                    f.unlink()
                    print(f"       Cleaned up external file: {f.name}")
                elif f.suffix == ".data":
                    f.unlink()
                    print(f"       Cleaned up external file: {f.name}")
            except Exception:
                pass


def convert(version: str):
    """Full conversion pipeline: download → export → embed weights."""
    if version not in MODELS:
        print(f"Error: Unknown version '{version}'. Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    export_dir = OUTPUT_DIR / version
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    print(f"Converting Parakeet TDT {version} to ONNX")
    print(f"Output directory: {export_dir}")
    print()

    model = download_model(version)

    # Use NeMo's built-in ONNX export — handles all components correctly
    print("[2/5] Exporting model to ONNX using NeMo native export...")
    onnx_path = str(export_dir / "model.onnx")
    try:
        model.export(onnx_path)
        print(f"       Exported to: {onnx_path}")
    except Exception as e:
        print(f"       NeMo export() failed: {e}")
        print("       Trying manual torch.onnx.export fallback...")
        import torch
        model.eval()
        model.freeze()
        try:
            model.export(
                output=onnx_path,
                check_trace=False,
            )
            print(f"       Exported via fallback: {onnx_path}")
        except Exception as e2:
            print(f"       Fallback also failed: {e2}")
            print("       Attempting component-level export...")
            try:
                encoder_path = str(export_dir / "encoder.onnx")
                model.encoder.export(encoder_path)
                print(f"       Encoder exported: {encoder_path}")
            except Exception as e3:
                print(f"       All export methods failed: {e3}")
                sys.exit(1)

    # Embed external weights into ONNX files (fixes 2MB encoder issue)
    embed_external_weights(export_dir)

    # Export vocabulary
    export_vocabulary(model, export_dir)

    # Copy outputs to flat output/ for GitHub Actions release
    for f in export_dir.iterdir():
        if f.is_file():
            dest = OUTPUT_DIR / f.name
            if dest != f:
                shutil.copy2(f, dest)

    print()
    print("Conversion complete!")
    print(f"Artifacts in: {export_dir}")
    print()
    print("Files:")
    for f in sorted(export_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA Parakeet TDT models to ONNX for VoiceMate Windows"
    )
    parser.add_argument(
        "--version", "-v",
        choices=list(MODELS.keys()),
        default="v2",
        help="Model version to convert (default: v2)",
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    convert(args.version)


if __name__ == "__main__":
    main()
