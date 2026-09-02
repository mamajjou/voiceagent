#!/bin/bash
set -e
# Build NeMo-Speech.cpp CUDA server
WORKDIR="${1:-/workspace/NeMo-Speech.cpp}"
if [ -d "$WORKDIR/.git" ]; then
  echo "NeMo-Speech.cpp already at $WORKDIR"
  cd "$WORKDIR"
  git pull
else
  echo "Cloning NeMo-Speech.cpp..."
  git clone https://github.com/NVIDIA/NeMo-Speech.cpp "$WORKDIR"
  cd "$WORKDIR"
fi
# Configure & build
scripts/configure.sh cuda-server || ./scripts/configure.sh cuda-server
cmake --build --preset cuda-server -j$(nproc)
echo "Build done. Binary at build/cuda-server/nemo-speech-server"
./build/cuda-server/nemo-speech-server --help 2>&1 | head -40 || true
# Diagnostic
./build/cuda-server/nemo-speech-server --diagnostic 2>&1 | head -100 || true
