# AMD / ROCm Adaptation Guide — RX 7900 XTX (amdbox)

This repo is a **fork of the NVIDIA RTX 3090 baseline** (`mamajjou/voiceagent`),
adapted to run on an **AMD Radeon RX 7900 XTX** (24GB GDDR6) via ROCm.

Everything downstream of the two inference servers is identical. Only the two
runtime builds change. Read the NVIDIA provenance notes in the git log /
`README.md` for the measured baseline you're porting.

## What's already measured on the 3090 (target to match)

- Nemotron 3.5 Streaming 0.6B Q8 ASR server: ~1270 MiB VRAM, WER ≈ 0, streaming.
- Qwen3.8-27B UD-Q4_K_M (llama.cpp CUDA): prompt ~69 tok/s, gen ~25–30 tok/s.
- E2E latency waterfall (EOU=650): speech_end → ASR final ~+800ms, → Qwen TTFT ~+1340ms.
- BOTH models co-resident: 16803 MiB / 24576 MiB total.

Migrate those numbers to AMD and re-measure — the goal is not that they match
exactly, but that you *know* them on the XTX.

## Key adaptations

### 1. llama.cpp → ROCm/HIP build (for Qwen)

The NVIDIA `llama-server` was built with `-DGGML_CUDA=ON`. On AMD use HIP:

```bash
# ROcm prerequisites (Ubuntu 24.04 + ROCm 6.x)
sudo apt install libamd-comgr-dev libhipblas-dev rocm-hip-sdk rocm-device-libs

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -S . -B build-hip \
  -DGGML_HIP=ON \
  -DGGML_HIPBLAS=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --target llama-server -j$(nproc)
```

- `gfx1100` = RX 7900 XTX. Confirm with `rocminfo | grep gfx`.
- If the model's `rms_norm`/attention fusions don't support HIP, unset
  `GGML_HIPBLAS` and use `GGML_HIP=ON` fallback (slower but works).

### 2. NeMo-Speech.cpp → HIP

NVIDIA ships a prebuilt CUDA release. On AMD you must build from source with the
HIP backend. Check `NeMo-Speech.cpp` `scripts/configure.sh` for a `hip`/`rocm`
preset (or `GGML_HIP=ON`). If the prebuilt is CUDA-only, build:

```bash
git clone https://github.com/NVIDIA/NeMo-Speech.cpp
cd NeMo-Speech.cpp
scripts/configure.sh hip-server   # verify preset name in repo
cmake --build --preset hip-server
```

If NeMo-Speech.cpp has **no** HIP backend yet, the cleanest AMD-safe fallback
that preserves the architecture is to run **Nemotron ASR via llama.cpp's own
`llama-server` with `--asr` / gguf-asr** OR swap the ASR server for a ROCm-capable
inference host — but keep the exact same WebSocket `/v1/realtime` contract so the
Python orchestrator (untouched) keeps working.

### 3. Config / path changes

```bash
# config/default.yaml
asr.backend.host/port:  keep 127.0.0.1:8090
llm.host/port:          keep 127.0.0.1:8081
# model paths already point at /workspace/models/... — keep
# Same 8GB: c 8192, np 1, fa on, ctk/ctv q8_0 — the XTX has 24GB like the 3090
```

### 4. VRAM budget (same as 3090)

- ASR (Nemotron 0.6B Q8): ~1.3 GB
- Qwen 27B Q4_K_M: ~15.5 GB
- Total co-resident: ~16.8 GB / 24 GB → ~7 GB free for KV growth.

Reference the startup scripts (`scripts/start_services.sh`) and swap the
server binaries for your ROCm builds.

## Testing

```bash
python -m voice_agent --source file --audio eval/sample.wav --language en-US
# and the E2E latency decomp harness (port it from the 3090 notes):
# speech_end -> ASR final -> Qwen TTFT waterfall
```

Measure: WER (EN+DE), EOU latency, Qwen prompt/gen tok/s, VRAM, and the
speech-end → first-token waterfall. Compare to the 3090 numbers above.

## Provenance

Mirrors `mamajjou/voiceagent` at commit `7429874`, pushed to this fork
(`voiceagent-amd`) for isolated AMD work. Do **not** push back to the NVIDIA
repo — keep the two baseline tracks separate.
