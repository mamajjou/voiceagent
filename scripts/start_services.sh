#!/bin/bash
set -e
# Start Nemotron and Qwen servers on single 3090
# Usage: ./scripts/start_services.sh [--qwen-only|--asr-only]

MODELS_DIR="/workspace/models"
QWEN_MODEL="$MODELS_DIR/Qwen3.8-27B-Q4_K_M.gguf"
# fallback names
if [ ! -f "$QWEN_MODEL" ]; then
  QWEN_MODEL=$(ls $MODELS_DIR/*Qwen*27B*.gguf 2>/dev/null | head -1 || echo "$MODELS_DIR/Qwen3.8-27B-Q4_K_M.gguf")
fi
NEMO_MODEL="$MODELS_DIR/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
CONFIG="config/default.yaml"

check_vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }

echo "=== Qwen llama-server ==="
echo "Model: $QWEN_MODEL"
echo "VRAM before: $(check_vram) MB"
if [[ "$1" != "--asr-only" ]]; then
  llama-server \
    -m "$QWEN_MODEL" \
    -ngl 999 \
    -c 8192 \
    -np 1 \
    -fa on \
    -ctk q8_0 \
    -ctv q8_0 \
    --host 127.0.0.1 \
    --port 8081 \
    --metrics \
    --verbose &
  QWEN_PID=$!
  echo "Qwen PID $QWEN_PID"
  sleep 5
  echo "VRAM after Qwen: $(check_vram) MB"
  curl -s http://127.0.0.1:8081/health | head -20 || curl -s http://127.0.0.1:8081/v1/models | head -20 || echo "Qwen not yet ready"
fi

echo "=== Nemotron NeMo-Speech.cpp ==="
if [[ "$1" != "--qwen-only" ]]; then
  NEMO_BIN="/workspace/NeMo-Speech.cpp/build/cuda-server/nemo-speech-server"
  if [ ! -f "$NEMO_BIN" ]; then
    echo "NeMo binary not found at $NEMO_BIN, run setup_nemo.sh"
    exit 1
  fi
  $NEMO_BIN \
    --model "$NEMO_MODEL" \
    --host 127.0.0.1 \
    --port 8090 \
    --gpu 0 \
    --rnnt-right-context 3 \
    --eou-ms 650 &
  ASR_PID=$!
  echo "ASR PID $ASR_PID"
  sleep 3
  echo "VRAM after ASR: $(check_vram) MB"
  curl -s http://127.0.0.1:8090/health | head -20 || echo "ASR health check"
fi

echo "VRAM final: $(check_vram) MB / 24576 MB"
echo "Logs: Qwen $QWEN_PID, ASR $ASR_PID"
wait
