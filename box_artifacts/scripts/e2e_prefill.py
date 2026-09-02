"""End-to-end prefill-while-speaking orchestrator.

While the user speaks, ASR emits partials. For each partial, we send a short
Qwen prefill that warms the KV cache for (system + history + current partial).
At EOU, we send the final text; llama.cpp reuses the cached prefix so most of
the prompt is already prefilled -> fast first token.

Measures the full speech_end -> first-token latency WITH prefill-while-speaking,
and compares it to the baseline (no prefill during speech).

Because llama.cpp /completion uses a raw prompt string (not chat history), we
build the prompt as a single string with a stable prefix. For chat with history,
the same applies as long as the serialized conversation is byte-identical up to
the growing user text.
"""
import asyncio, json, time, os, sys
import numpy as np
import soundfile as sf
import websockets
import httpx

SR=16000; CHUNK_MS=20; cs=SR*CHUNK_MS//1000
ASR_WS='ws://127.0.0.1:8090/v1/realtime'
BASE='http://127.0.0.1:8081'
MODEL='/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM="You are a helpful assistant. Answer concisely."
REPLY_TAIL="\nAssistant:"

def load_pcm16(path):
    data,sr=sf.read(path,always_2d=False,dtype='float32')
    if data.ndim==2: data=data.mean(axis=1)
    if sr!=SR:
        dur=len(data)/sr; tgt=int(dur*SR)
        old=np.linspace(0,1,len(data)); new=np.linspace(0,1,tgt)
        data=np.interp(new,old,data).astype(np.float32)
    data=np.clip(data,-1,1); return (data*32767).astype(np.int16)

def speech_end(pcm,thr=0.01):
    n=len(pcm); nch=(n+cs-1)//cs; last=0
    for i in range(nch):
        s=i*cs; e=min(s+cs,n); c=pcm[s:e].astype(np.float32)/32768.0
        if c.size and np.sqrt(np.mean(c**2))>thr: last=i+1
    return min(last*cs,n)

async def qwen_prefill(prompt):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            await client.post(f"{BASE}/completion", json={
                "model":MODEL,"prompt":prompt,"max_tokens":1,"cache_prompt":True,"temperature":0.7})
        except Exception:
            pass

async def qwen_generate(prompt, t0, on_delta):
    async with httpx.AsyncClient(timeout=60.0) as client:
        t_ref=time.monotonic(); first=None; full=""
        async with client.stream("POST", f"{BASE}/completion", json={
            "model":MODEL,"prompt":prompt,"max_tokens":64,"cache_prompt":True,"temperature":0.7}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    d=line[5:].strip()
                    if d and d!="[DONE]":
                        try:
                            j=json.loads(d)
                            tok=j.get("content") or ""
                            if tok:
                                if first is None: first=time.monotonic()-t0
                                full+=tok
                        except: pass
    return first, full

async def run(path, lang, eou_ms, use_prefill):
    pcm=load_pcm16(path); total=len(pcm)
    se=speech_end(pcm)/SR; speech_chunks=int((se*SR)//cs)
    silent=np.zeros(cs,dtype=np.int16)
    commit_wall=time.monotonic()+se+eou_ms/1000.0
    t0=time.monotonic(); final=''; partials_ts=[]
    async with websockets.connect(ASR_WS,max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(),2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{'sample_rate':SR,'language':lang}}))
        # track partial text; stable prefix for prefill
        stable_prefix=[SYSTEM+"\nUser: "]
        last_prefill_len=[0]
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
                    await ws.send(json.dumps({'type':'input_audio_buffer.commit'})); break
                tgt=min(t0+speech_chunks*CHUNK_MS/1000.0+(j*CHUNK_MS/1000.0),commit_wall)
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes()); j+=1
            await asyncio.sleep(0.1)
        async def receiver():
            nonlocal final
            accum=''
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),12.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg); tt=d.get('type','')
                if tt=='conversation.item.input_audio_transcription.delta':
                    dl=d.get('delta') or d.get('text') or ''
                    if dl:
                        accum+=dl
                        partials_ts.append(time.monotonic()-t0)
                        if use_prefill:
                            # fire prefill in background so it doesn't block the receiver
                            asyncio.create_task(qwen_prefill(stable_prefix[0]+accum+REPLY_TAIL))
                            last_prefill_len[0]=len(accum)
                elif tt=='conversation.item.input_audio_transcription.completed':
                    final=d.get('transcript') or d.get('text') or ''; break
                elif tt=='error': break
        await asyncio.gather(sender(), receiver())
    # at EOU, generate using the full prompt (warm if prefill else cold)
    if use_prefill:
        # allow background prefills to settle before generating
        await asyncio.sleep(0.2)
    full_prompt=SYSTEM+"\nUser: "+final+REPLY_TAIL
    first, full = await qwen_generate(full_prompt, t0, None)
    return {'se':se,'asr_final_after_se':(partials_ts[-1]-se) if partials_ts else None,
            'ttft_after_se':(first-se) if first else None,'text':final[:40],'llm':full[:40],'n_partials':len(partials_ts)}

async def main():
    clips=[('/tmp/en3.wav','en-US'),('/tmp/de4.wav','de-DE'),('/tmp/en2.wav','en-US')]
    print("=== E2E PREFILL-WHILE-SPEAKING vs BASELINE (speech_end -> first-token) ===\n")
    print(f"{'clip':<10} {'mode':<8} {'ASR_after_SE':>12} {'TTFT_after_SE':>14} {'saved':>8}")
    for path,lang in clips:
        row={}
        for use,label in [(False,'baseline'),(True,'prefill ')]:
            r=await run(path,lang,650,use)
            row[label]=r
            af=r['asr_final_after_se']
            tt=r['ttft_after_se']
            af_s=f"{af*1000:.0f}" if af is not None else "-"
            tt_s=f"{tt*1000:.0f}" if tt is not None else "-"
            print(f"{os.path.basename(path):<10} {label:<8} {af_s:>12} {tt_s:>14}  {r['text']!r}")
        # saved
        if 'baseline' in row and 'prefill ' in row and row['baseline']['ttft_after_se'] and row['prefill ']['ttft_after_se']:
            print(f"  -> SAVED {(row['baseline']['ttft_after_se']-row['prefill ']['ttft_after_se'])*1000:.0f} ms")
        print()

asyncio.run(main())
