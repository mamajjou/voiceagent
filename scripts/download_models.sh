#!/bin/bash
set -e
# Download Nemotron 3.5 Q8 and Qwen3.8-27B Q4_K_M
MODELS_DIR="${1:-/workspace/models}"
mkdir -p "$MODELS_DIR"
echo "Models dir: $MODELS_DIR"
echo "VRAM check: 24GB RTX 3090, Qwen 17.77GB + Nemotron 0.74GB + KV ~1-2GB"

# Check hf cli
if ! command -v hf &>/dev/null; then
  pip install -q huggingface_hub
fi
echo "Downloading Nemotron 3.5 Q8 (742 MB)..."
hf download nvidia/nemotron-3.5-asr-streaming-0.6b nemotron-3.5-asr-streaming-0.6b.q8_0.gguf --local-dir "$MODELS_DIR" || \
  wget -c -P "$MODELS_DIR" https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/main/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf || echo "Nemotron download failed - manual required"

echo "Downloading Qwen 27B Q4_K_M (~17.77 GB)... this will take a while"
# Try several Qwen quants - Q4_K_M is preferred
hf download Qwen/Qwen3-8-27B-GGUF qwen3-27b-q4_k_m.gguf --local-dir "$MODELS_DIR" 2>&1 | head -20 || \
  echo "Try: hf download bartowski/Qwen3-27B-GGUF --local-dir $MODELS_DIR"

# Optional Sortformer
echo "Optional Sortformer Q8 (147 MB)..."
hf download nvidia/diar_streaming_sortformer_4spk-v2 diar_streaming_sortformer_4spk-v2.q8_0.gguf --local-dir "$MODELS_DIR" || echo "Sortformer optional, skipping"

ls -lh "$MODELS_DIR" || true
nvidia-smi || true
