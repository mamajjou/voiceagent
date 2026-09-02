# voiceagent — Naive Real-Time Voice Baseline

`audio → Nemotron 3.5 Streaming 0.6B → UTF-8 → Qwen3.8-27B` on one RTX 3090 24GB.

> How quickly can we get a stable transcript + first Qwen token after the human stops speaking, without annoying premature cuts?

## Repo as backup for volatile VM

Develop on `ssh box` (`/workspace`), sync back and push:

```bash
# dev on box in /workspace/voiceagent
ssh box
cd /workspace/voiceagent
# ... work ...

# from host, save (excludes weights)
./scripts/pull-from-box.sh
git add -A && git commit -m "feat: ..." && git push
# or one-liner:
./scripts/save.sh "feat: endpoint sweep"
```

Weights (`*.gguf`, `*.safetensors`, etc.) are excluded from rsync — keep them only on box's `/workspace/models`.

## Quickstart (box)

```bash
# 1. models ~18GB total
./scripts/download_models.sh /workspace/models

# 2. build NeMo-Speech.cpp
./scripts/setup_nemo.sh

# 3. start both servers (Qwen + Nemotron) — one 3090
./scripts/start_services.sh
# Qwen: http://127.0.0.1:8081  Nemotron: ws://127.0.0.1:8090

# 4. test mock pipeline (no GPU needed)
python -m voice_agent --source file --audio eval/sample.wav --mock --language en-US

# 5. full pipeline (needs servers)
python -m voice_agent --source file --audio eval/audio/ami_...wav --language en-US
python -m voice_agent --source mic --language de-DE

# 6. eval
python eval/prepare_manifest.py --ami 3 --voxpopuli-de 50
python eval/replay.py --manifest eval/manifest.jsonl --realtime 0
python eval/sweep_endpointing.py --mock   # 15 configs
python eval/report.py  # -> REPORT.md
```

## Hardware

- RTX 3090 24GB, CUDA 12/13, Ubuntu 24.04
- Qwen3.8-27B Q4_K_M ≈17.77GB GGUF, context 8192, q8 KV, FA on
- Nemotron 3.5 Q8 ≈742MB, Sortformer Q8 ≈147MB optional

If VRAM tight: 8192→4096, verify q8 KV, remove Sortformer, then smaller quant.

## Architecture

```
AudioSource (FileReplay / Microphone)  20ms PCM16 16kHz
    ↓ websocket
NeMo-Speech.cpp (Nemotron)  rnnt_right_context 3 (320ms) + EOU 650ms
    ↓ UTF-8 final only
TurnManager (IDLE → LISTENING → LLM_GENERATING)
    ↓ OpenAI SSE
llama-server (Qwen3.8)  thinking off, cache_prompt true
    ↓ streamed text
terminal + JSONL events + VRAM telemetry
```

Partial ASR `~` is display-only; only `✓` final goes to Qwen.

## Config

`config/default.yaml` — ASR language `en-US`/`de-DE`, no auto-detect for bench.

## Evaluation

- AMI `diarizers-community/ami` 2-5 recordings
- VoxPopuli `facebook/voxpopuli` de 50-100 utterances
- Manifest `eval/manifest.jsonl` canonical input
- Metrics: WER, false endpoint rate, endpoint/ASR/TTFT latency (median/p90/p95), partial stability

## Timing waterfall

Each `runs/<id>/events.jsonl`:

```json
{"t":3.924,"event":"asr_final","text":"..."}
{"t":3.928,"event":"llm_request"}
{"t":4.143,"event":"llm_first_token"}
```

Decomposition: lookahead + endpoint + finalize + Qwen prefill + first decode.

## Layout

```
config/  pyproject.toml  src/voice_agent/  eval/  scripts/  tests/  runs/
```

## License

MIT — eval data via respective dataset licenses.
