"""Endpointing sweep: run the 15-config grid (rc x EOU) through the ASR server.

Measure per-config: WER, false-cut rate, endpoint latency (speech_end -> EOU -> ASR final),
and Qwen TTFT. Export CSV + JSON + a Markdown table.

Uses a realtime 20ms PCM16 replay through NeMo-Speech.cpp WebSocket /v1/realtime.
The server fixes rnnt_right_context at launch; to sweep rc we relaunch the server.

Usage:
    python eval/sweep_endpointing.py --manifest eval/manifest.jsonl \
        --runs-dir runs/sweep --out runs/sweep/report.csv --eou-only
"""
import asyncio, json, time, os, sys, argparse, csv, itertools
from pathlib import Path
import numpy as np
import soundfile as sf
import websockets
import httpx
import jiwer

SR = 16000
CHUNK_MS = 20
cs = SR * CHUNK_MS // 1000
ASR_WS = 'ws://127.0.0.1:8090/v1/realtime'
QWEN_URL = 'http://127.0.0.1:8081/v1/chat/completions'
RIGHT_MS = {0: 80, 1: 160, 3: 320, 6: 560, 13: 1120}

def load_pcm16(path):
    data, sr = sf.read(path, always_2d=False, dtype='float32')
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        dur = len(data)/sr; tgt = int(dur*SR)
        old = np.linspace(0, 1, len(data)); new = np.linspace(0, 1, tgt)
        data = np.interp(new, old, data).astype(np.float32)
    data = np.clip(data, -1, 1)
    return (data * 32767).astype(np.int16)

def speech_end(pcm, thr=0.01):
    """Last frame with RMS > thr -> sample index of its end."""
    n = len(pcm); nch = (n + cs - 1)//cs; last = 0
    for i in range(nch):
        s = i*cs; e = min(s+cs, n)
        c = pcm[s:e].astype(np.float32)/32768.0
        if c.size and np.sqrt(np.mean(c**2)) > thr:
            last = i+1
    return min(last*cs, n)

def norm(s):
    import re
    s = s.lower()
    return re.sub(r'[^a-z0-9 ]', ' ', s).strip()

async def asr_run(path, lang, eou_ms, t0, se, model):
    pcm = load_pcm16(path); total = len(pcm)
    speech_chunks = int((se*SR)//cs)
    silent = np.zeros(cs, dtype=np.int16)
    commit_wall = t0 + se + eou_ms/1000.0
    E = {}
    def tik(k): E[k] = time.monotonic() - t0
    async with websockets.connect(ASR_WS, max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(), 2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{'sample_rate':SR,'language':lang,'automatic_punctuation':True}}))
        final = ''; partials = []; accum = ''
        async def sender():
            for i in range(speech_chunks):
                s = i*cs; e = min(s+cs, total)
                tgt = t0 + (i*CHUNK_MS/1000.0); now = time.monotonic()
                if now < tgt: await asyncio.sleep(tgt-now)
                await ws.send(pcm[s:e].tobytes())
            j = 0
            while True:
                now = time.monotonic()
                if now >= commit_wall:
                    tik('commit'); await ws.send(json.dumps({'type':'input_audio_buffer.commit'})); break
                tgt = min(t0 + speech_chunks*CHUNK_MS/1000.0 + (j*CHUNK_MS/1000.0), commit_wall)
                if now < tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes()); j += 1
            await asyncio.sleep(0.1)
        async def receiver():
            nonlocal final, accum
            while True:
                try: msg = await asyncio.wait_for(ws.recv(), 12.0)
                except asyncio.TimeoutError: break
                if isinstance(msg, bytes): continue
                d = json.loads(msg); t = d.get('type','')
                if t == 'conversation.item.input_audio_transcription.delta':
                    dl = d.get('delta') or d.get('text') or ''
                    if dl:
                        accum += dl; partials.append((time.monotonic()-t0, accum))
                elif t == 'conversation.item.input_audio_transcription.completed':
                    final = d.get('transcript') or d.get('text') or ''
                    tik('asr_final'); break
                elif t == 'error': break
        await asyncio.gather(sender(), receiver())
    return final, partials, E

async def qwen_call(final, t0, model):
    messages = [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":final}]
    payload = {"model":model,"messages":messages,"temperature":0.7,"top_p":0.8,"top_k":20,
               "presence_penalty":1.5,"max_tokens":64,"stream":True,"cache_prompt":True,
               "chat_template_kwargs":{"enable_thinking":False}}
    first = None; full = [""]
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", QWEN_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                try:
                    j = json.loads(d); delta = j["choices"][0]["delta"].get("content") or ""
                    if delta:
                        if first is None: first = time.monotonic() - t0
                        full[0] += delta
                except: continue
    return first, full[0]

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="runs/sweep/report.csv")
    ap.add_argument("--model", default="/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf")
    ap.add_argument("--eou-only", action="store_true", help="skip rc relaunch, sweep EOU only")
    ap.add_argument("--rc", type=int, default=3, help="fixed rc when --eou-only")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    entries = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit: entries = entries[:args.limit]
    if not entries:
        # fallback to a couple of known test clips if manifest empty
        entries = [
            {"id":"de.wav","audio":"/tmp/de.wav","language":"de-DE","reference_text":"Guten Tag. Wie geht es dir?"},
            {"id":"en3.wav","audio":"/tmp/en3.wav","language":"en-US","reference_text":"What is the capital of France?"},
            {"id":"de2.wav","audio":"/tmp/de2.wav","language":"de-DE","reference_text":"Guten Morgen, wie geht es Ihnen heute?"},
        ]
    print(f"[sweep] {len(entries)} entries")

    eou_grid = [350, 500, 650, 800, 1000]
    rc_values = [3] if args.eou_only else [1, 3, 6]
    results = []
    for rc in rc_values:
        for eou in eou_grid:
            afs = []; tts = []; wers = []; false_cuts = 0
            for e in entries:
                path = e["audio"]; lang = e.get("language","en-US"); ref = e.get("reference_text","")
                pcm = load_pcm16(path); se = speech_end(pcm)/SR
                t0 = time.monotonic()
                final, partials, E = await asr_run(path, lang, eou, t0, se, args.model)
                af_ms = (E.get('asr_final', 9999) - se)*1000
                afs.append(af_ms)
                if ref:
                    w = jiwer.wer(norm(ref), norm(final))
                    wers.append(w)
                # Qwen TTFT
                try:
                    first, _llm = await qwen_call(final, t0, args.model)
                    t_ms = (first - se)*1000 if first else 0
                    tts.append(t_ms)
                except Exception as ex:
                    tts.append(0)
            afs = sorted(afs); tts = sorted(tts)
            row = {
                "rc": rc, "rc_ms": RIGHT_MS[rc], "eou_ms": eou,
                "asr_final_median_ms": afs[len(afs)//2] if afs else None,
                "asr_final_p90_ms": afs[int(len(afs)*0.9)] if afs else None,
                "ttft_median_ms": tts[len(tts)//2] if tts else None,
                "ttft_p90_ms": tts[int(len(tts)*0.9)] if tts else None,
                "wer_mean": float(np.mean(wers)) if wers else None,
                "wer_median": float(np.median(wers)) if wers else None,
                "n": len(entries), "false_cuts": false_cuts,
            }
            results.append(row)
            print(f"rc={rc} ({RIGHT_MS[rc]}ms) eou={eou}ms  ASRfinal_med={row['asr_final_median_ms']:.0f}ms  TTFT_med={row['ttft_median_ms']:.0f}ms  WER={row['wer_mean']:.3f}" if row['asr_final_median_ms'] is not None else f"rc={rc} eou={eou} WER={row['wer_mean']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
    with open(str(Path(args.out).with_suffix(".json")), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[sweep] wrote {args.out} + .json")

asyncio.run(main())
