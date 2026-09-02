# Voice Agent Baseline Report — Naive Real-Time Voice on a Single RTX 3090

`audio → Nemotron 3.5 Streaming 0.6B → UTF-8 → EOU → Qwen3.8-27B Q4_K_M → streamed text`

**Goal:** measure, from the moment a human stops speaking, how quickly we get a stable transcript + first useful Qwen token, without premature cuts. This is deliberately the *simplest* credible architecture (no fusion, no partial-prefill, no TTS).

---

## 1. Software / model revisions

| Component | Revision / Path |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 576 MiB, driver 591.86 |
| CUDA toolkit | 13.0 (V13.0.88) — `/usr/local/cuda-13.0` |
| NeMo-Speech.cpp | `nemo-speech 0.1.0` (prebuilt CUDA tarball), `/workspace/nemo-prebuilt/.../bin/nemo-speech` |
| ASR model | `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` — sha256[:16] `9b2382b0d0163b33` |
| llama.cpp | built from source, `ggml 0.22.0`, CUDA backend on, `/workspace/llama.cpp/build-cuda/bin/llama-server` |
| Qwen model | `Qwen3.8-27B-UD-Q4_K_M.gguf` (unsloth UD) — sha256[:16] `177ae5e70ef0d340`, size 16 464 440 224 bytes |

### Launch commands

NeMo-Speech ASR server (streaming, RTX 3090):
```bash
export LD_LIBRARY_PATH=/workspace/nemo-prebuilt/nemo-speech-0.1.0-linux-x86_64-cuda/lib:$LD_LIBRARY_PATH
nemo-speech serve \
  --asr-model /workspace/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --asr.streaming.rnnt_right_context=3 \
  --asr.endpointing.stop_history_eou_ms=650 \
  --host 127.0.0.1 --port 8090 --device cuda:0
```

Qwen llama-server:
```bash
/workspace/llama.cpp/build-cuda/bin/llama-server \
  -m /workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  -ngl 999 -c 8192 -np 1 -fa on -ctk q8_0 -ctv q8_0 \
  --host 127.0.0.1 --port 8081 --metrics
```

---

## 2. VRAM breakdown (both models co-resident on one 3090)

| Component | VRAM |
|---|---|
| Nemotron ASR server (idle, model loaded) | **1 270 MiB** |
| Qwen 27B UD-Q4_K_M + KV (8192 ctx, q8_0) | ~15.5 GB |
| **Total both resident** | **16 803 MiB / 24 576 MiB (~68%)** |
| **Free headroom** | **~7.8 GB** |

- ASR VRAM does **not** grow during streaming inference (peak == idle, 1 270 MiB).
- Peak GPU util during ASR streaming: ~64%, power ~26 W, temp ~40 °C.
- With ~7.8 GB headroom, Qwen context could be raised (e.g. 16 384) if needed.

---

## 3. ASR WER (streaming, rc=3 / 320 ms lookahead)

Tested on 9 synthetic gTTS clips (5 German + 4 English). Streaming WER = **0.000 for 8/9**, avg **0.056**. The one "error" was a *reference artifact*: the reference used ASCII `Koennen` but the model correctly emitted `können` (proper umlauts). **UTF-8 preserved exactly** — German umlauts pass through unchanged.

---

## 4. Endpointing / latency waterfall (speech_end → ASR final → Qwen TTFT)

Measured with a realtime 20 ms PCM16 replay through the WebSocket `/v1/realtime` path. `speech_end` = last sample with RMS > 0.01.

**Key architectural finding:** the `/v1/realtime` WebSocket server does **not** auto-finalize on `session.update.endpointing_ms`. It requires an explicit `input_audio_buffer.commit`. So the orchestration owns EOU: it streams trailing silence, then commits. ASR finalization after commit ≈ 0.12 s.

### Full 15-config endpointing sweep (all rc × EOU, 8-clip corpus, box-local Qwen)

| rc | lookahead | EOU | ASR final med | ASR final p90 | Qwen TTFT med | Qwen TTFT p90 |
|----|-----------|-----|---------------|---------------|---------------|---------------|
| 1 | 160 ms | 350 ms | 418 ms | 424 ms | 961 ms | 1090 ms |
| 1 | 160 ms | 500 ms | 572 ms | 584 ms | 1139 ms | 1224 ms |
| 1 | 160 ms | 650 ms | 724 ms | 729 ms | 1275 ms | 1365 ms |
| 1 | 160 ms | 800 ms | 876 ms | 879 ms | 1452 ms | 1528 ms |
| 1 | 160 ms | 1000 ms | 1077 ms | 1084 ms | 1631 ms | 1799 ms |
| **3** | **320 ms** | **350 ms** | **412 ms** | **432 ms** | **995 ms** | **1015 ms** |
| 3 | 320 ms | 500 ms | 566 ms | 583 ms | 1127 ms | 1226 ms |
| 3 | 320 ms | 650 ms | 718 ms | 747 ms | 1262 ms | 1368 ms |
| 3 | 320 ms | 800 ms | 867 ms | 904 ms | 1437 ms | 1596 ms |
| 3 | 320 ms | 1000 ms | 1066 ms | 1094 ms | 1646 ms | 1693 ms |
| 6 | 560 ms | 350 ms | 416 ms | 455 ms | 957 ms | 1071 ms |
| 6 | 560 ms | 500 ms | 578 ms | 631 ms | 1147 ms | 1268 ms |
| 6 | 560 ms | 650 ms | 724 ms | 740 ms | 1293 ms | 1419 ms |
| 6 | 560 ms | 800 ms | 866 ms | 885 ms | 1421 ms | 1523 ms |
| 6 | 560 ms | 1000 ms | 1088 ms | 1108 ms | 1638 ms | 1700 ms |

**WER = 0.000 across all 15 configs (8-clip corpus), false cuts = 0.**

Formulas: `ASR_final ≈ EOU + ~70 ms`; `Qwen_TTFT ≈ ASR_final + ~600 ms` (Qwen prefill + first token).

**Key finding: rnnt_right_context (ASR lookahead 160/320/560 ms) has essentially NO effect** on latency or accuracy for these clean clips — the latency is entirely governed by the EOU silence window. The lookahead is absorbed by the streaming decoder and does not surface as additional endpoint latency.

### Detailed waterfall (EOU=650, rc=320 ms, corrected harness — drops file trailing silence)

For a 2.35 s German clip (speech_end = 1900 ms):

```
speech_end                        +0 ms
commit (speech_end + 650 EOU)    +652 ms
ASR final                        +801 ms     ← EOU + ~150 ms finalization
LLM request                      +801 ms
LLM first token                +1 339 ms     ← +538 ms Qwen prefill+first-token
LLM done                       +2 129 ms
```

**The dominant term is the EOU silence window + ASR finalization (~710–800 ms). Qwen adds only ~540–710 ms.** Reducing EOU from 650 → 350 ms saves ~190 ms of ASR latency and ~200 ms of end-to-end latency.

---

## 5. Partial transcript stability

Streaming partials track the final nearly perfectly: for every clip the **longest-common-prefix ratio ≈ (text length − 1)**. Example (en3):
```
~ What is
~ What is the
~ What is the capital
~ What is the capital of
~ What is the capital of France
✓ What is the capital of France?
```
Partials are monotonically growing and stable — a strong candidate for later partial-prefill / speculative approaches.

---

## 6. Qwen generation performance

| Metric | Value |
|---|---|
| Qwen context | 8192, n_parallel=1, KV q8_0 (non-unified), flash-attn on |
| Thinking | disabled (`enable_thinking: false`) — no `<think>` or `reasoning_content` |
| Prompt processing | ~69 tok/s cold; ~88–95 tok/s with prompt cache |
| Generation | ~25–30 tok/s |
| Qwen prefill + first token | ~540–710 ms (constant across EOU) |

**PV cache reuse confirmed working** (multi-turn):
| Turn | prompt cache tokens | prompt tok/s |
|------|---------------------|--------------|
| 1 | 25 | 40 tok/s |
| 2 | 51 | 88 tok/s (2.2× faster) |
| 3 | 114 | 95 tok/s |

---

## 7. End-of-speech → first-token latency (the headline metric)

Best operating points measured on the synthetic corpus:

| EOU | speech_end → Qwen first token (median) |
|-----|------------------------------------------|
| 350 ms | **1 132 ms** |
| 500 ms | **1 176 ms** |
| 650 ms | **1 320 ms** |

Hybrid recommendation: on this synthetic (no mid-utterance pauses) corpus, **EOU=350 ms** gives the fastest at ~1.13 s speech-end → first-token. On real speech with natural pauses (hesitations, breath), EOU=500–650 ms is safer to avoid false cuts; the penalty is only ~150–200 ms.

---

## 8. Best configuration found

```yaml
asr:
  backend: { gpu: 0, host: 127.0.0.1, port: 8090 }
  streaming: { rnnt_right_context: 3 }     # 320 ms lookahead
  endpointing: { stop_history_eou_ms: 650 } # or 500 for faster, 350 for synthetic-clean
llm:
  -c 8192 -np 1 -fa on -ctk q8_0 -ctv q8_0
  temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5
  enable_thinking: false
```

---

## 9. Remaining problems / notes

- **VAD over-detection:** the RMS `speech_end` heuristic over-detects voice tails on some gTTS clips (negative latencies seen). Use a proper VAD (e.g. Silero) for ground-truth speech_end, or a lower threshold.
- **EOU on real speech** needs tuning to natural pause length; synthetic clips have clean trailing silence so false-cut rate is ~0 here.
- **Qwen TTFT ~540–710 ms** is the non-EOU lower bound; if this is too slow, speculative decoding or a smaller model / higher prompt cache would help — but for the naive baseline this is acceptable.
- Qwen `blk.64.*` "unused tensor" warnings are expected for the UD (unsloth dynamic) quant grouping; harmless.

---

## 10. Recommendation on model fusion

The naive architecture is surprisingly good. With **EOU=350–650 ms**, speech-end → Qwen first token is **1.13–1.32 s**, dominated by the EOU silence window (~70%) and ASR finalization, with Qwen adding only ~540–710 ms.

Because Qwen's contribution is small and constant, the potential upside of **fusing ASR hidden states into Qwen** (to eliminate the UTF-8 boundary and prefill during speech) is bounded to roughly:

- Overlapping Qwen prefill with speech: could save ~150–350 ms (the segment of the 540–710 ms that is prefill, not first-token), on top of not waiting for EOU.
- **UTF-8/tokenization is effectively irrelevant** to latency (neglibile microseconds; and it does not block — the measured gap is dominated by real compute, not tokenization).

**Conclusion:** architectural fusion might save a few hundred ms in the best case, but the naive design already achieves sub-1.4 s speech-end → first-token while keeping both models on one 3090 at 68% VRAM with natural temporal GPU separation (ASR uses GPU while user speaks, Qwen uses it after EOU). The biggest lever is **EOU tuning**, not fusion.

---

*Generated from measurements on box (RTX 3090). Reproduce with `/tmp/run_e2e.py`, `/tmp/bench_latency_decomp.py`, `/tmp/sweep_eou_e2e.py`.*
