#!/usr/bin/env python3
"""End-to-end voice pipeline runner on the box 3090.

audio (FileReplay, 20ms PCM16) -> Nemotron streaming ASR (ws /v1/realtime)
    -> EOU commit -> Qwen (llama-server) -> streamed text.

Uses box-local services: ASR ws://127.0.0.1:8090, Qwen http://127.0.0.1:8081.
Reports per-turn latency waterfall and writes runs/<id>/events.jsonl.

Usage: python eval/e2e_pipeline.py --audio /tmp/de.wav --language de-DE [--eou 650]
"""
import asyncio, json, time, os, sys, argparse, uuid
import numpy as np
import soundfile as sf
import websockets
import httpx

SR = 16000
CHUNK_MS = 20
cs = SR * CHUNK_MS // 1000
ASR_WS = 'ws://127.0.0.1:8090/v1/realtime'
QWEN_URL = 'http://127.0.0.1:8081/v1/chat/completions'
SYSTEM_PROMPT = ("You are participating in a live spoken conversation. "
                 "Respond naturally and directly. Prefer concise conversational answers. "
                 "The user's message was produced by automatic speech recognition. "
                 "Resolve obvious minor transcription errors from context when possible. "
                 "Do not mention the speech recognition system.")

def load_pcm16(path):
    data, sr = sf.read(path, always_2d=False, dtype='float32')
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        dur = len(data)/sr; tgt = int(dur*SR)
        old = np.linspace(0,1,len(data)); new = np.linspace(0,1,tgt)
        data = np.interp(new, old, data).astype(np.float32)
    data = np.clip(data, -1, 1)
    return (data * 32767).astype(np.int16)

def speech_end(pcm, thr=0.01):
    n = len(pcm); nch=(n+cs-1)//cs; last=0
    for i in range(nch):
        s=i*cs; e=min(s+cs,n); c=pcm[s:e].astype(np.float32)/32768.0
        if c.size and np.sqrt(np.mean(c**2))>thr: last=i+1
    return min(last*cs, n)

async def run_turn(path, lang, eou_ms, model_path, log_events=None):
    pcm = load_pcm16(path); total=len(pcm)
    se = speech_end(pcm)/SR
    speech_chunks = int((se*SR)//cs)
    silent = np.zeros(cs, dtype=np.int16)
    commit_wall = time.monotonic() + se + eou_ms/1000.0
    t0 = time.monotonic()
    evs = []
    def ev(name, **kw):
        t=time.monotonic()-t0
        e={"t":round(t,3),"event":name,**kw}
        evs.append(e)
        if log_events is not None: log_events.append(e)
    ev("audio_start", audio=os.path.basename(path), language=lang)
    ev("reference_speech_end", se=round(se,3))

    async with websockets.connect(ASR_WS, max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(), 2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{'sample_rate':SR,'language':lang,'automatic_punctuation':True}}))
        final=''; partials=[]
        async def sender():
            for i in range(speech_chunks):
                s=i*cs; e=min(s+cs,total)
                tgt=t0+(i*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(pcm[s:e].tobytes())
            j=0
            while True:
                now=time.monotonic()
                if now>=commit_wall:
                    ev("eou_commit"); await ws.send(json.dumps({'type':'input_audio_buffer.commit'})); break
                tgt=min(t0+speech_chunks*CHUNK_MS/1000.0+(j*CHUNK_MS/1000.0), commit_wall)
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes()); j+=1
            await asyncio.sleep(0.1)
        async def receiver():
            nonlocal final
            accum=''
            while True:
                try: msg=await asyncio.wait_for(ws.recv(), 12.0)
                except asyncio.TimeoutError: break
                if isinstance(msg, bytes): continue
                d=json.loads(msg); t=d.get('type','')
                if t=='conversation.item.input_audio_transcription.delta':
                    dl=d.get('delta') or d.get('text') or ''
                    if dl:
                        accum+=dl; partials.append((time.monotonic()-t0, accum))
                        ev("asr_partial", text=accum)
                elif t=='conversation.item.input_audio_transcription.completed':
                    final=d.get('transcript') or d.get('text') or ''
                    ev("asr_final", text=final); break
                elif t=='error': break
        await asyncio.gather(sender(), receiver())

    ev("llm_request")
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":final}]
    payload={"model":model_path,"messages":messages,"temperature":0.7,"top_p":0.8,"top_k":20,
             "presence_penalty":1.5,"max_tokens":96,"stream":True,"cache_prompt":True,
             "chat_template_kwargs":{"enable_thinking":False}}
    full=[""]; first=None
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", QWEN_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"): continue
                d=line[5:].strip()
                if d=="[DONE]": break
                try:
                    j=json.loads(d); delta=j["choices"][0]["delta"].get("content") or ""
                    if delta:
                        if first is None: first=time.monotonic()-t0; ev("llm_first_token", text=delta)
                        full[0]+=delta
                except: continue
    ev("llm_done", text=full[0][:200])

    return {
        "path": os.path.basename(path), "language": lang,
        "speech_end_s": round(se,3), "duration_s": round(total/SR,3),
        "asr_text": final, "partials": len(partials),
        "asr_final_after_se_ms": round(([e for e in evs if e["event"]=="asr_final"][0]["t"] - se)*1000,1) if any(e["event"]=="asr_final" for e in evs) else None,
        "llm_ttft_after_se_ms": round((first - se)*1000,1) if first else None,
        "qwen_text": full[0][:160],
        "events": evs,
    }

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="/tmp/de.wav")
    ap.add_argument("--language", default="de-DE")
    ap.add_argument("--eou", type=int, default=650)
    ap.add_argument("--model", default="/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf")
    ap.add_argument("--runs-dir", default="runs/e2e")
    args = ap.parse_args()
    run_id = str(uuid.uuid4())[:8]
    log_dir = f"{args.runs_dir}/{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    log_events=[]
    r = await run_turn(args.audio, args.language, args.eou, args.model, log_events)
    with open(f"{log_dir}/events.jsonl","w") as f:
        for e in log_events: f.write(json.dumps(e, ensure_ascii=False)+"\n")
    with open(f"{log_dir}/summary.json","w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\n[run] {run_id} -> {log_dir}")

asyncio.run(main())
